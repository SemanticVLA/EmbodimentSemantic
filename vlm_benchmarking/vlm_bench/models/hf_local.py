import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, BitsAndBytesConfig

from .base import BaseVLM


class HFLocalVLM(BaseVLM):
    """Generic HuggingFace transformers fallback for models unsupported by vLLM."""

    kind = "local"

    def _load(self, model_path: str, **kwargs):
        # vLLM injects its own config classes into the global transformers registry.
        # We must scrub the registry of any vLLM-injected configs so AutoModel uses 
        # the official remote code from the model repository instead.
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if hasattr(CONFIG_MAPPING, "_extra_content"):
            to_remove = [k for k, v in CONFIG_MAPPING._extra_content.items() if getattr(v, "__module__", "").startswith("vllm")]
            for k in to_remove:
                del CONFIG_MAPPING._extra_content[k]

        dtype = getattr(torch, kwargs.get("dtype", "bfloat16"))
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except ValueError:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        quant_kwargs = {}
        if kwargs.get("load_in_4bit") or kwargs.get("load_in_8bit"):
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=kwargs.get("load_in_4bit", False),
                load_in_8bit=kwargs.get("load_in_8bit", False),
            )

        load_kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True, **quant_kwargs)
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

    def query(self, image: Image.Image, prompt: str) -> str:
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature is not None and self.temperature > 0:
            gen_kwargs.update(temperature=self.temperature, do_sample=True)

        # Isaac uses string content with "<image>" as a literal token placeholder,
        # separate messages per item, and tensor_stream for generation.
        if getattr(self.processor, "vision_token", None):
            messages = [
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": prompt},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            device = next(self.model.parameters()).device
            with torch.no_grad():
                output_ids = self.model.generate(
                    tensor_stream=inputs["tensor_stream"].to(device), **gen_kwargs
                )
            return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Standard transformers path
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def query_batch(self, images: list, prompt: str) -> list:
        return [self.query(img, prompt) for img in images]

    def close(self):
        del self.model
        torch.cuda.empty_cache()

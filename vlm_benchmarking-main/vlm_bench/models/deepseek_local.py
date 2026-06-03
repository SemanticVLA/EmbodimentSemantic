import torch
from PIL import Image

from .base import BaseVLM


class DeepseekLocalVLM(BaseVLM):
    """Native DeepSeek-VL2 loader.

    DeepSeek-VL2 is not loaded reliably through the generic HuggingFace
    AutoProcessor/AutoConfig path used by hf_local.py. Use the model package's
    own processor/model classes and generation flow instead.
    """

    kind = "local"

    def _load(self, model_path: str, **kwargs):
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

        dtype_name = kwargs.get("dtype", "bfloat16")
        self.dtype = getattr(torch, dtype_name)
        self.chunk_size = kwargs.get("chunk_size", 512)

        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.dtype).cuda().eval()

    def query(self, image: Image.Image, prompt: str) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": ["image"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt="",
        ).to(self.model.device)

        gen_kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        if self.temperature is not None and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            if self.chunk_size:
                inputs_embeds, past_key_values = self.model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=self.chunk_size,
                )
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,
                    **gen_kwargs,
                )
                generated = outputs[0][len(prepare_inputs.input_ids[0]):]
            else:
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    **gen_kwargs,
                )
                generated = outputs[0]

        return self.tokenizer.decode(
            generated.cpu().tolist(),
            skip_special_tokens=True,
        ).strip()

    def query_batch(self, images: list[Image.Image], prompt: str) -> list[str]:
        return [self.query(img, prompt) for img in images]

    def close(self):
        del self.model
        torch.cuda.empty_cache()

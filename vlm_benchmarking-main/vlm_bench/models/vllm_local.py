import traceback
from PIL import Image
from .base import BaseVLM

_FALLBACK_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  ⚠  vLLM LOAD FAILED — falling back to HuggingFace transformers ║
║  Model : {model:<52} ║
║  Reason: {reason:<52} ║
╚══════════════════════════════════════════════════════════════════╝
"""


class VllmVLM(BaseVLM):
    """Single vLLM-backed adapter for all supported local VLM families.
    Falls back to a generic HuggingFace transformers backend if vLLM
    cannot load the model (e.g. unsupported architecture).
    """

    kind = "local"

    def _load(self, model_path: str, **kwargs):
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.max_pixels = kwargs.get("max_pixels", None)

        quantization = kwargs.get("quantization", None)
        if quantization is None:
            if kwargs.get("load_in_4bit") or kwargs.get("load_in_8bit"):
                quantization = "bitsandbytes"

        llm_kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            max_model_len=kwargs.get("max_model_len", 8192),
            limit_mm_per_prompt={"image": 1},
            dtype=kwargs.get("dtype", "bfloat16"),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
        )
        if quantization:
            llm_kwargs["quantization"] = quantization

        # Pass pixel limit to the vision encoder at engine-init time (Qwen models).
        # This is the correct vLLM API — not image_kwargs inside multi_modal_data.
        if self.max_pixels is not None:
            llm_kwargs["mm_processor_kwargs"] = {"max_pixels": self.max_pixels}

        try:
            self.llm = LLM(**llm_kwargs)
            self._fallback = None
        except Exception as e:
            reason = str(e).splitlines()[0][:52]
            print(_FALLBACK_BANNER.format(model=model_path[:52], reason=reason), flush=True)
            print("=" * 68, flush=True)
            from .hf_local import HFLocalVLM
            kwargs["trust_remote_code"] = True
            self._fallback = HFLocalVLM(model_path, **kwargs)
            self._fallback.registry_name = getattr(self, "registry_name", "")

    def _make_sampling_params(self):
        if self.temperature is not None and self.temperature > 0:
            return self._SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        # temperature=0 requires top_k=1; vLLM rejects temperature=0 + top_p=1.0
        return self._SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_k=1,
        )

    def query(self, image: Image.Image, prompt: str) -> str:
        if self._fallback is not None:
            return self._fallback.query(image, prompt)
        return self.query_batch([image], prompt)[0]

    def query_batch(self, images: list[Image.Image], prompt: str) -> list[str]:
        if self._fallback is not None:
            return self._fallback.query_batch(images, prompt)

        tokenizer = self.llm.get_tokenizer()
        sampling_params = self._make_sampling_params()

        inputs = []
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img}})

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def close(self):
        if self._fallback is not None:
            self._fallback.close()
            return
        import torch
        # Shut down the vLLM engine and its background workers before releasing.
        try:
            self.llm.llm_engine.shutdown()
        except Exception:
            pass
        del self.llm
        torch.cuda.empty_cache()

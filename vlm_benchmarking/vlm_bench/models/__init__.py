from .base import BaseVLM

# Maps the "kind" field in models.yaml to the adapter class.
# To support a new provider, add one line here and a new adapter file.
_KIND_TO_CLASS = {
    "gemini":    ("vlm_bench.models.gemini",     "GeminiVLM"),
    "openai":    ("vlm_bench.models.openai",     "OpenAIVLM"),
    "anthropic": ("vlm_bench.models.anthropic",  "AnthropicVLM"),
    "nvidia":    ("vlm_bench.models.nvidia",     "NvidiaVLM"),
    "vllm":      ("vlm_bench.models.vllm_local", "VllmVLM"),
    "local":     ("vlm_bench.models.hf_local",   "HFLocalVLM"),
}


def build_model(name: str, cfg: dict) -> BaseVLM:
    cfg = dict(cfg)  # don't mutate the caller's config
    kind = cfg.pop("kind", None)
    if kind is None:
        raise ValueError(f"Model '{name}' in models.yaml is missing a 'kind' field. "
                         f"Expected one of: {list(_KIND_TO_CLASS)}")
    if kind not in _KIND_TO_CLASS:
        raise ValueError(f"Unknown kind '{kind}' for model '{name}'. "
                         f"Expected one of: {list(_KIND_TO_CLASS)}")

    import importlib
    module_path, cls_name = _KIND_TO_CLASS[kind]
    cls = getattr(importlib.import_module(module_path), cls_name)

    model_path = cfg.pop("model_id")
    instance = cls(model_path, **cfg)
    instance.registry_name = name
    return instance


__all__ = ["BaseVLM", "build_model"]

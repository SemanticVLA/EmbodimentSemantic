from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class BaseVLM(ABC):
    kind: str = ""        # "local" or "api"
    registry_name: str = ""  # set by build_model()

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.temperature = kwargs.get("temperature", None)
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.batch_size = kwargs.get("batch_size", 1)
        self._load(model_path, **kwargs)

    @abstractmethod
    def _load(self, model_path: str, **kwargs):
        """Load model weights / initialise API client."""

    @abstractmethod
    def query(self, image: Image.Image, prompt: str) -> str:
        """Run inference on a single PIL image and return the response string."""

    def query_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Run inference on a list of images with the same prompt.
        Override in subclasses for true batched GPU inference; default falls back to query()."""
        return [self.query(img, prompt) for img in images]

    def close(self):
        """Release resources. Local adapters override to free GPU memory."""

    def __repr__(self):
        return f"{self.__class__.__name__}(registry_name={self.registry_name!r})"

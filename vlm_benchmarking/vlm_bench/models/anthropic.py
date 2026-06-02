import base64
import io
import os
import time

import anthropic
from PIL import Image

from .base import BaseVLM


class AnthropicVLM(BaseVLM):
    kind = "api"

    def _load(self, model_path: str, **kwargs):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = kwargs.get("max_retries", 5)

    def query(self, image: Image.Image, prompt: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        for attempt in range(self.max_retries):
            try:
                extra = {}
                if self.temperature is not None:
                    extra["temperature"] = self.temperature
                response = self.client.messages.create(
                    model=self.model_path,
                    max_tokens=self.max_new_tokens,
                    **extra,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def close(self):
        if hasattr(self, "client"):
            self.client.close()

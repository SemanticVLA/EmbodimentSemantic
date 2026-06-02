import base64
import io
import os
import time

from openai import OpenAI, RateLimitError
from PIL import Image

from .base import BaseVLM


class OpenAIVLM(BaseVLM):
    kind = "api"

    def _load(self, model_path: str, **kwargs):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.max_retries = kwargs.get("max_retries", 5)
        self.reasoning_effort = kwargs.get("reasoning_effort")
        self.verbosity = kwargs.get("verbosity")

    def _is_gpt5_family(self) -> bool:
        return self.model_path.startswith("gpt-5")

    def query(self, image: Image.Image, prompt: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.model_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                }

                if self._is_gpt5_family():
                    request_kwargs["max_completion_tokens"] = self.max_new_tokens
                    if self.reasoning_effort is not None:
                        request_kwargs["reasoning_effort"] = self.reasoning_effort
                    if self.verbosity is not None:
                        request_kwargs["verbosity"] = self.verbosity
                else:
                    request_kwargs["temperature"] = self.temperature
                    request_kwargs["max_tokens"] = self.max_new_tokens

                response = self.client.chat.completions.create(**request_kwargs)
                return response.choices[0].message.content.strip()
            except RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def close(self):
        if hasattr(self, "client"):
            self.client.close()

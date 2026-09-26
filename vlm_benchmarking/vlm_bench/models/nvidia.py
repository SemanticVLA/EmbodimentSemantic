import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests
from PIL import Image

from .base import BaseVLM

_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_TIMEOUT = 480  # seconds — agentview images are large and slow under load
_DEFAULT_MAX_RETRIES = 8
_MAX_RETRY_WAIT = 120



# Models that reject top_p / frequency_penalty / presence_penalty
_STRICT_PARAM_MODELS = {
    "microsoft/phi-4-multimodal-instruct",
}


class NvidiaVLM(BaseVLM):
    kind = "api"

    def _load(self, model_path: str, **kwargs):
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise EnvironmentError("NVIDIA_API_KEY environment variable not set")
        self._api_key = api_key
        self.max_retries = kwargs.get("max_retries", _DEFAULT_MAX_RETRIES)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def _retry_wait(self, attempt: int, is_502: bool = False) -> int:
        base = 10 if is_502 else 5
        return min(base * (2 ** attempt), _MAX_RETRY_WAIT)

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _build_payload(self, b64: str, prompt: str) -> dict:
        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.model_path not in _STRICT_PARAM_MODELS:
            payload["top_p"] = self.top_p
            payload["frequency_penalty"] = self.frequency_penalty
            payload["presence_penalty"] = self.presence_penalty
        return payload

    def _query_one(self, image: Image.Image, prompt: str) -> str:
        b64 = self._encode_image(image)
        payload = self._build_payload(b64, prompt)

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    _NVIDIA_API_URL, headers=self._headers, json=payload, timeout=_TIMEOUT
                )

                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        is_502 = resp.status_code == 502
                        wait = self._retry_wait(attempt, is_502=is_502)
                        print(f"[Attempt {attempt + 1}/{self.max_retries}] Retrying in {wait}s due to status {resp.status_code}")
                        time.sleep(wait)
                        continue

                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait = self._retry_wait(attempt, is_502=True)
                    print(f"[Attempt {attempt + 1}/{self.max_retries}] Retrying in {wait}s due to: {type(e).__name__}")
                    time.sleep(wait)
                else:
                    raise
            except requests.HTTPError:
                raise

        return ""

    def query(self, image: Image.Image, prompt: str) -> str:
        return self._query_one(image, prompt)

    def query_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Fire all images concurrently; results returned in original order."""
        results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            futures = {
                pool.submit(self._query_one, img, prompt): i
                for i, img in enumerate(images)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[query_batch] frame {idx} failed after all retries: {e}")
                    results[idx] = ""
        return results

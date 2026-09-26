import ast
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from google import genai
from google.genai import types
from PIL import Image

from .base import BaseVLM

_BASE_BACKOFF = 60
_DEFAULT_MAX_RETRIES = 8
_MAX_RETRY_WAIT = 120


def _extract_error_payload(msg: str) -> dict:
    start = msg.find("{")
    if start < 0:
        return {}
    try:
        payload = ast.literal_eval(msg[start:])
    except (SyntaxError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _summarize_gemini_error(error: Exception) -> str:
    msg = str(error).strip()
    payload = _extract_error_payload(msg)
    err = payload.get("error", {}) if isinstance(payload.get("error"), dict) else {}

    code = err.get("code") or getattr(error, "code", None)
    if code is None:
        match = re.search(r"\b([45]\d\d)\b", msg)
        code = match.group(1) if match else None

    status = err.get("status") or getattr(error, "status", None)
    if status is None:
        match = re.search(r"\b([A-Z_]{4,})\b", msg)
        status = match.group(1) if match else None

    message = err.get("message") or getattr(error, "message", None)
    if not message:
        message = msg
    message = " ".join(str(message).split())

    quota_bits = []
    retry_delay = None
    for detail in err.get("details", []):
        if not isinstance(detail, dict):
            continue
        if "retryDelay" in detail:
            retry_delay = detail["retryDelay"]
        for violation in detail.get("violations", []):
            if not isinstance(violation, dict):
                continue
            metric = violation.get("quotaMetric")
            quota_id = violation.get("quotaId")
            quota_value = violation.get("quotaValue")
            if metric:
                quota_bits.append(f"quotaMetric={metric}")
            if quota_id:
                quota_bits.append(f"quotaId={quota_id}")
            if quota_value:
                quota_bits.append(f"quotaValue={quota_value}")

    parts = [type(error).__name__]
    if code is not None:
        parts.append(str(code))
    if status:
        parts.append(str(status))
    if retry_delay:
        parts.append(f"retryDelay={retry_delay}")
    if quota_bits:
        parts.extend(quota_bits[:3])
    if message:
        parts.append(message[:350])
    return " | ".join(parts)


class GeminiVLM(BaseVLM):
    kind = "api"

    def _load(self, model_path: str, **kwargs):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.max_retries = kwargs.get("max_retries", _DEFAULT_MAX_RETRIES)
        thinking_budget = kwargs.get("thinking_budget", 1024)
        self._gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_new_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            seed=42,
        )

    def _image_part(self, image: Image.Image) -> types.Part:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return types.Part.from_bytes(data=buf.read(), mime_type="image/png")

    def _retry_wait(self, attempt: int, is_gateway: bool = False) -> int:
        base = 10 if is_gateway else 5
        return min(base * (2 ** attempt), _MAX_RETRY_WAIT)

    def _query_one(self, image: Image.Image, prompt: str) -> str:
        img_part = self._image_part(image)
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_path,
                    contents=[img_part, types.Part.from_text(text=prompt)],
                    config=self._gen_config,
                )
                return response.text or ""
            except Exception as e:
                msg = str(e)
                is_gateway = "502" in msg or "504" in msg
                should_retry = (
                    "429" in msg
                    or "500" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "timeout" in msg.lower()
                    or "connection" in msg.lower()
                )

                if should_retry and attempt < self.max_retries - 1:
                    if "429" in msg:
                        match = re.search(r"retryDelay.*?(\d+)s", msg)
                        wait = min(int(match.group(1)) + 5, _MAX_RETRY_WAIT) if match else _BASE_BACKOFF * (attempt + 1)
                    else:
                        wait = self._retry_wait(attempt, is_gateway=is_gateway)

                    print(
                        f"[Attempt {attempt + 1}/{self.max_retries}] "
                        f"Retrying in {wait}s due to: {_summarize_gemini_error(e)}",
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    raise
        return ""

    def query(self, image: Image.Image, prompt: str) -> str:
        return self._query_one(image, prompt)

    def query_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Run independent Gemini requests concurrently; preserve input order."""
        if len(images) <= 1 or self.batch_size <= 1:
            return [self._query_one(img, prompt) for img in images]

        results = [None] * len(images)
        max_workers = min(self.batch_size, len(images))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._query_one, img, prompt): i
                for i, img in enumerate(images)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[query_batch] frame {idx} failed after all retries: {_summarize_gemini_error(e)}")
                    results[idx] = ""
        return results

    def close(self):
        if hasattr(self, "client"):
            self.client.close()

"""Minimal provider-neutral LLM interface and DeepSeek implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Protocol
from urllib import error, request


class LLM(Protocol):
    """The only capability KernelBunny needs from a model provider."""

    def generate(self, prompt: str) -> str:
        """Return generated text for one prompt."""


class ProviderError(RuntimeError):
    """The provider could not return a usable response."""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_seconds: float = 120.0
    max_retries: int = 1
    max_tokens: int = 8_000


class DeepSeekLLM:
    """Small OpenAI-compatible JSON-mode client for DeepSeek."""

    def __init__(self, config: DeepSeekConfig | None = None) -> None:
        self.config = config or DeepSeekConfig()

    def generate(self, prompt: str) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Missing API key in environment variable {self.config.api_key_env}."
            )

        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "max_tokens": self.config.max_tokens,
            }
        ).encode("utf-8")

        last_error = "Unknown provider failure."
        for attempt in range(self.config.max_retries + 1):
            http_request = request.Request(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(
                    http_request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _response_text(payload)
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")[-2_000:]
                last_error = f"DeepSeek returned HTTP {exc.code}: {details}"
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"DeepSeek request failed: {exc}"
            except (KeyError, TypeError, ValueError) as exc:
                last_error = f"DeepSeek returned an unexpected response: {exc}"

            if attempt < self.config.max_retries:
                time.sleep(2**attempt)

        raise ProviderError(last_error)


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting existing environment."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise TypeError("top-level response is not an object")
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("response contains no choices")
    message = choices[0]["message"]
    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason", "unknown")
        raise ValueError(
            f"response content is empty (finish_reason={finish_reason})"
        )
    return content

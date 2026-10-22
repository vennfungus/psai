"""Provider adapter for OpenAI-compatible chat-completions APIs."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol, Sequence

import httpx

from .config import RAGConfig


class LLMClient(Protocol):
    """Provider-neutral interface consumed by RAGGenerator."""

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str: ...


class OpenAICompatibleClient:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self._owns_client = http_client is None

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API credentials: set the {self.config.api_key_env} "
                "environment variable for the configured LLM endpoint."
            )
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout_seconds)
            )

        base_url = self.config.base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        try:
            response = self._http_client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"LLM request timed out after {self.config.timeout_seconds:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LLM endpoint returned HTTP {exc.response.status_code}; "
                "check the endpoint, model, and provider account."
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not reach the configured LLM endpoint: {exc}"
            ) from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "LLM endpoint returned an unexpected chat-completions response"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM endpoint returned an empty answer")
        return content.strip()

    def close(self) -> None:
        """Close an HTTP client owned by this adapter."""
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

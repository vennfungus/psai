"""Lazy, offline-only text generation using Hugging Face Transformers."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Protocol

from .base import GenerationResult, TextGenerationClient
from .config import GenerationOptions, LocalTransformerConfig
from .exceptions import (
    ContextLengthError,
    InferenceError,
    InsufficientResourcesError,
    LocalGenerationError,
    ModelNotFoundError,
    UnsupportedConfigurationError,
)


class _Backend(Protocol):
    context_length: int | None
    device: str

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompt: str, options: GenerationOptions) -> str: ...


class _TransformersBackend:
    def __init__(self, config: LocalTransformerConfig) -> None:
        if not config.model_path.is_dir():
            raise ModelNotFoundError(
                f"Local generative model directory does not exist: "
                f"{config.model_path}"
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise UnsupportedConfigurationError(
                "Local generation requires PyTorch and transformers. "
                "Install the project requirements first."
            ) from exc

        self.device = self._resolve_device(
            config.device,
            torch,
            quantization=config.quantization,
        )
        if config.quantization == "dynamic-int8" and self.device != "cpu":
            raise UnsupportedConfigurationError(
                "dynamic-int8 quantization is supported on CPU only"
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(config.model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(config.model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self.context_length = _model_context_length(self.model.config)
            if self.tokenizer.eos_token_id is None:
                raise UnsupportedConfigurationError(
                    "The local model tokenizer must define an end-of-sequence token"
                )
            if config.quantization == "dynamic-int8":
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
        except (ModelNotFoundError, UnsupportedConfigurationError):
            raise
        except MemoryError as exc:
            raise InsufficientResourcesError(
                f"Insufficient memory to load model at {config.model_path}"
            ) from exc
        except OSError as exc:
            raise ModelNotFoundError(
                f"Required local model files are unavailable at "
                f"{config.model_path}: {exc}"
            ) from exc
        except Exception as exc:
            if _is_out_of_memory(exc):
                raise InsufficientResourcesError(
                    f"Insufficient memory to load model at {config.model_path}: {exc}"
                ) from exc
            raise InferenceError(
                f"Could not load local Transformer model at {config.model_path}: {exc}"
            ) from exc
        self.torch = torch

    @staticmethod
    def _resolve_device(
        device: str,
        torch: Any,
        *,
        quantization: str,
    ) -> str:
        if device == "auto":
            if quantization == "dynamic-int8":
                return "cpu"
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
            if callable(mps) and mps():
                return "mps"
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise UnsupportedConfigurationError(
                "CUDA was requested but is not available in this PyTorch installation"
            )
        mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
        if device == "mps" and not (callable(mps) and mps()):
            raise UnsupportedConfigurationError(
                "MPS was requested but is not available in this PyTorch installation"
            )
        return device

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def generate(self, prompt: str, options: GenerationOptions) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }
            generation_args: dict[str, Any] = {
                "max_new_tokens": options.max_new_tokens,
                "do_sample": options.temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if options.temperature > 0:
                generation_args["temperature"] = options.temperature
                generation_args["top_p"] = options.top_p
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **generation_args)
            generated = output[0, inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
        except MemoryError as exc:
            raise InsufficientResourcesError(
                "Insufficient memory for this generation request"
            ) from exc
        except Exception as exc:
            if _is_out_of_memory(exc):
                raise InsufficientResourcesError(
                    f"Insufficient memory for this generation request: {exc}"
                ) from exc
            raise InferenceError(f"Local Transformer inference failed: {exc}") from exc


def _model_context_length(model_config: Any) -> int | None:
    for attribute in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
    ):
        value = getattr(model_config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _is_out_of_memory(error: Exception) -> bool:
    return "out of memory" in str(error).casefold()


class LocalTransformerClient(TextGenerationClient):
    """Load one local causal Transformer on first use and reuse it thereafter."""

    def __init__(
        self,
        config: LocalTransformerConfig,
        *,
        backend: _Backend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend
        self._backend_lock = Lock()

    @property
    def context_length(self) -> int | None:
        return self._get_backend().context_length

    @property
    def device(self) -> str:
        return self._get_backend().device

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            return self._get_backend().count_tokens(text)
        except LocalGenerationError:
            raise
        except Exception as exc:
            raise InferenceError(f"Unable to count tokens: {exc}") from exc

    def generate(
        self,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        requested = options or GenerationOptions()
        requested.validate()
        resolved = GenerationOptions(
            max_new_tokens=(
                requested.max_new_tokens
                if requested.max_new_tokens is not None
                else self.config.max_new_tokens
            ),
            temperature=(
                requested.temperature
                if requested.temperature is not None
                else self.config.temperature
            ),
            top_p=(
                requested.top_p
                if requested.top_p is not None
                else self.config.top_p
            ),
        )
        resolved.validate()
        if resolved.max_new_tokens > self.config.max_new_tokens:
            raise UnsupportedConfigurationError(
                "Per-request max_new_tokens cannot exceed the configured limit"
            )
        backend = self._get_backend()
        input_tokens = self.count_tokens(prompt)
        model_context = backend.context_length or self.config.context_length
        context_limit = min(self.config.context_length, model_context)
        if input_tokens + resolved.max_new_tokens > context_limit:
            raise ContextLengthError(
                f"Prompt and output exceed model context: {input_tokens} input "
                f"tokens + {resolved.max_new_tokens} output tokens > "
                f"{context_limit}"
            )
        start = time.perf_counter()
        try:
            text = backend.generate(prompt, resolved)
        except LocalGenerationError:
            raise
        except Exception as exc:
            if _is_out_of_memory(exc):
                raise InsufficientResourcesError(
                    f"Insufficient memory for this generation request: {exc}"
                ) from exc
            raise InferenceError(f"Local Transformer inference failed: {exc}") from exc
        elapsed = time.perf_counter() - start
        if not isinstance(text, str) or not text.strip():
            raise InferenceError("Local Transformer returned an empty response")
        text = text.strip()
        output_tokens = self.count_tokens(text)
        return GenerationResult(
            text=text,
            metadata={
                "runtime": self.config.runtime,
                "model_path": str(self.config.model_path),
                "device": backend.device,
                "quantization": self.config.quantization,
                "context_length": context_limit,
                "max_new_tokens": resolved.max_new_tokens,
                "temperature": resolved.temperature,
                "top_p": resolved.top_p,
                "input_tokens": input_tokens,
                "generated_tokens": output_tokens,
                "generation_seconds": elapsed,
                "tokens_per_second": (
                    output_tokens / elapsed if elapsed > 0 else None
                ),
                "network_access": False,
            },
        )

    def _get_backend(self) -> _Backend:
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    self._backend = _TransformersBackend(self.config)
        return self._backend

"""Typed configuration for strictly local Transformer inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .exceptions import UnsupportedConfigurationError

RuntimeName = Literal["transformers"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
QuantizationName = Literal["none", "dynamic-int8"]


@dataclass(frozen=True)
class LocalTransformerConfig:
    model_path: Path
    runtime: RuntimeName = "transformers"
    device: DeviceName = "auto"
    quantization: QuantizationName = "none"
    context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.runtime != "transformers":
            raise UnsupportedConfigurationError(
                "runtime must be 'transformers'"
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise UnsupportedConfigurationError(
                "device must be one of: auto, cpu, cuda, mps"
            )
        if self.quantization not in {"none", "dynamic-int8"}:
            raise UnsupportedConfigurationError(
                "quantization must be 'none' or 'dynamic-int8'"
            )
        if self.context_length < 64:
            raise UnsupportedConfigurationError("context_length must be >= 64")
        if not 1 <= self.max_new_tokens < self.context_length:
            raise UnsupportedConfigurationError(
                "max_new_tokens must be positive and less than context_length"
            )
        if not 0 <= self.temperature <= 2:
            raise UnsupportedConfigurationError(
                "temperature must be between 0 and 2"
            )
        if not 0 < self.top_p <= 1:
            raise UnsupportedConfigurationError("top_p must be in (0, 1]")
        if self.quantization == "dynamic-int8" and self.device not in {
            "auto",
            "cpu",
        }:
            raise UnsupportedConfigurationError(
                "dynamic-int8 quantization is supported on CPU only"
            )


@dataclass(frozen=True)
class GenerationOptions:
    """Per-request overrides; omitted values use the client configuration."""

    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    def validate(self) -> None:
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise UnsupportedConfigurationError("max_new_tokens must be >= 1")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise UnsupportedConfigurationError(
                "temperature must be between 0 and 2"
            )
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise UnsupportedConfigurationError("top_p must be in (0, 1]")

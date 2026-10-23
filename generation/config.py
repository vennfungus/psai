"""Configuration for strictly local Transformer text generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalGenerationConfig:
    model_path: Path
    device: str = "cpu"
    context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0
    quantization: str = "none"

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.context_length < 64:
            raise ValueError("context_length must be >= 64")
        if self.max_new_tokens < 1 or self.max_new_tokens >= self.context_length:
            raise ValueError(
                "max_new_tokens must be positive and less than context_length"
            )
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.quantization not in {"none", "dynamic-int8"}:
            raise ValueError("quantization must be 'none' or 'dynamic-int8'")
        if self.quantization == "dynamic-int8" and self.device != "cpu":
            raise ValueError("dynamic-int8 quantization is supported on CPU only")

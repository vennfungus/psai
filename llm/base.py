"""Stable interface and result types for text-generation clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from .config import GenerationOptions

MetadataValue = str | int | float | bool | None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metadata: Mapping[str, MetadataValue]


class TextGenerationClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Generate text for a prompt and return it with runtime metadata."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens using the loaded model's tokenizer."""

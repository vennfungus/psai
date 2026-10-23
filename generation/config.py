"""Backward-compatible configuration name for local generation."""

from llm.config import LocalTransformerConfig

LocalGenerationConfig = LocalTransformerConfig

__all__ = ["LocalGenerationConfig"]

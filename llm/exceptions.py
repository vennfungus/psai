"""Explicit errors raised by the local text-generation runtime."""


class LocalGenerationError(Exception):
    """Base class for local generation failures."""


class ModelNotFoundError(LocalGenerationError, FileNotFoundError):
    """The configured model directory or required model files are missing."""


class UnsupportedConfigurationError(LocalGenerationError, ValueError):
    """The runtime, device, quantisation, or generation options are unsupported."""


class InsufficientResourcesError(LocalGenerationError, MemoryError):
    """The selected model or inference request exceeds available resources."""


class InferenceError(LocalGenerationError):
    """Model loading or text generation failed."""


class ContextLengthError(InferenceError):
    """The prompt and requested output exceed the model context window."""

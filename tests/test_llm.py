"""Offline tests for the local Transformer client and its public interface."""

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm.local_client import _TransformersBackend
from llm import (
    GenerationOptions,
    GenerationResult,
    ContextLengthError,
    InferenceError,
    InsufficientResourcesError,
    LocalTransformerClient,
    LocalTransformerConfig,
    ModelNotFoundError,
    TextGenerationClient,
    UnsupportedConfigurationError,
)


class MockBackend:
    context_length = 1024
    device = "cpu"

    def __init__(self, response: str = "Generated response.") -> None:
        self.response = response
        self.calls: list[tuple[str, GenerationOptions]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, options: GenerationOptions) -> str:
        self.calls.append((prompt, options))
        return self.response


class FailingBackend(MockBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def generate(self, prompt: str, options: GenerationOptions) -> str:
        raise self.error


class TextGenerationInterfaceTests(unittest.TestCase):
    def test_interface_is_abstract_and_result_is_typed(self) -> None:
        with self.assertRaises(TypeError):
            TextGenerationClient()
        result = GenerationResult("text", {"network_access": False})
        self.assertEqual(result.text, "text")
        self.assertFalse(result.metadata["network_access"])


class LocalTransformerConfigTests(unittest.TestCase):
    def test_defaults_and_path_normalisation(self) -> None:
        config = LocalTransformerConfig("models/local")
        self.assertEqual(config.model_path, Path("models/local"))
        self.assertEqual(config.runtime, "transformers")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_new_tokens, 512)

    def test_invalid_runtime_device_context_and_quantisation_are_rejected(self) -> None:
        invalid_values = (
            ({"runtime": "vllm"}, "runtime must be"),
            ({"device": "tpu"}, "device must be"),
            ({"context_length": 32}, "context_length"),
            ({"context_length": 64, "max_new_tokens": 64}, "max_new_tokens"),
            ({"quantization": "4bit"}, "quantization must be"),
            ({"device": "cuda", "quantization": "dynamic-int8"}, "CPU only"),
        )
        for overrides, message in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(UnsupportedConfigurationError, message):
                    LocalTransformerConfig(Path("model"), **overrides)

    def test_generation_options_are_validated(self) -> None:
        for options in (
            GenerationOptions(max_new_tokens=0),
            GenerationOptions(temperature=2.1),
            GenerationOptions(top_p=0),
        ):
            with self.subTest(options=options):
                with self.assertRaises(UnsupportedConfigurationError):
                    options.validate()


class LocalTransformerClientTests(unittest.TestCase):
    def test_transformers_load_is_local_only(self) -> None:
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        model = MagicMock()
        model.config.max_position_embeddings = 128
        model.to.return_value = model
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        auto_model = MagicMock()
        auto_model.from_pretrained.return_value = model
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = auto_tokenizer
        transformers.AutoModelForCausalLM = auto_model

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"transformers": transformers}):
                _TransformersBackend(
                    LocalTransformerConfig(Path(directory), device="cpu")
                )

        for loader in (
            auto_tokenizer.from_pretrained,
            auto_model.from_pretrained,
        ):
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])

    def test_mock_inference_returns_text_and_useful_metadata(self) -> None:
        backend = MockBackend()
        client = LocalTransformerClient(
            LocalTransformerConfig(Path("local-model")),
            backend=backend,
        )

        result = client.generate("A short prompt.")

        self.assertEqual(result.text, "Generated response.")
        self.assertEqual(result.metadata["runtime"], "transformers")
        self.assertEqual(result.metadata["device"], "cpu")
        self.assertEqual(result.metadata["max_new_tokens"], 512)
        self.assertFalse(result.metadata["network_access"])
        self.assertEqual(result.metadata["input_tokens"], 3)
        self.assertEqual(len(backend.calls), 1)

    def test_backend_is_loaded_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = MockBackend()
            config = LocalTransformerConfig(Path(directory))
            with patch(
                "llm.local_client._TransformersBackend",
                return_value=backend,
            ) as backend_factory:
                client = LocalTransformerClient(config)
                client.generate("first request")
                client.generate("second request")

        backend_factory.assert_called_once_with(config)
        self.assertEqual(len(backend.calls), 2)

    def test_missing_model_directory_raises_specific_error(self) -> None:
        client = LocalTransformerClient(
            LocalTransformerConfig(Path("missing-model"))
        )
        with self.assertRaises(ModelNotFoundError):
            client.generate("A prompt")

    def test_prompt_context_and_request_limits_are_enforced(self) -> None:
        backend = MockBackend()
        backend.context_length = 8
        client = LocalTransformerClient(
            LocalTransformerConfig(
                Path("local-model"),
                context_length=128,
                max_new_tokens=4,
            ),
            backend=backend,
        )
        with self.assertRaisesRegex(ContextLengthError, "exceed model context"):
            client.generate("one two three four five")
        with self.assertRaises(UnsupportedConfigurationError):
            client.generate(
                "prompt",
                GenerationOptions(max_new_tokens=5),
            )

    def test_inference_and_resource_errors_are_explicit(self) -> None:
        for error, expected in (
            (RuntimeError("device operation failed"), InferenceError),
            (MemoryError("out of memory"), InsufficientResourcesError),
        ):
            with self.subTest(error=error):
                client = LocalTransformerClient(
                    LocalTransformerConfig(Path("local-model")),
                    backend=FailingBackend(error),
                )
                with self.assertRaises(expected):
                    client.generate("A prompt")

    def test_empty_prompt_and_empty_model_response_are_rejected(self) -> None:
        client = LocalTransformerClient(
            LocalTransformerConfig(Path("local-model")),
            backend=MockBackend(" "),
        )
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            client.generate(" ")
        with self.assertRaises(InferenceError):
            client.generate("A prompt")


if __name__ == "__main__":
    unittest.main()

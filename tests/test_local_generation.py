"""Offline tests for the local-only text generation adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from generation import LocalGenerationConfig, LocalTransformerGenerator
from llm import GenerationOptions, GenerationResult


class MockRuntime:
    def __init__(self, response: str, token_count: int = 5) -> None:
        self.response = response
        self.token_count = token_count
        self.last_prompt: str | None = None

    def count_tokens(self, text: str) -> int:
        return self.token_count

    def generate(
        self,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        self.last_prompt = prompt
        return GenerationResult(self.response, {})


class LocalGenerationTests(unittest.TestCase):
    def test_grounded_prompt_keeps_retrieval_data_untrusted_and_citations(self) -> None:
        runtime = MockRuntime("Task T-1 is required [S1].")
        result = LocalTransformerGenerator(
            LocalGenerationConfig(Path("not-loaded"), max_new_tokens=8),
            runtime=runtime,
        ).generate(
            "What is required?",
            [{"source_ref": "S1", "text": "Task T-1 is required."}],
        )
        self.assertIn("untrusted quoted data", runtime.last_prompt or "")
        self.assertIn("source_refs", runtime.last_prompt or "")
        self.assertEqual(result["answer"], "Task T-1 is required [S1].")
        self.assertFalse(result["model_metadata"]["network_access"])

    def test_unknown_citations_are_warned(self) -> None:
        runtime = MockRuntime("Unsupported claim [S99].")
        result = LocalTransformerGenerator(
            LocalGenerationConfig(Path("local"), max_new_tokens=8),
            runtime=runtime,
        ).generate("Question?", [{"source_ref": "S1", "text": "Evidence"}])
        self.assertTrue(any("S99" in warning for warning in result["warnings"]))

    def test_context_overflow_fails_without_silent_truncation(self) -> None:
        runtime = MockRuntime("answer", token_count=20)
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(
                Path("local"),
                context_length=64,
                max_new_tokens=50,
            ),
            runtime=runtime,
        )
        with self.assertRaisesRegex(ValueError, "exceeds the configured model context"):
            generator.generate(
                "Question?",
                [{"source_ref": "S1", "text": "Evidence"}],
            )
        self.assertIsNone(runtime.last_prompt)

    def test_structured_output_must_be_valid_json(self) -> None:
        runtime = MockRuntime("{invalid")
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(Path("local"), max_new_tokens=8),
            runtime=runtime,
        )
        with self.assertRaisesRegex(RuntimeError, "valid JSON"):
            generator.generate(
                "Create a plan",
                [{"source_ref": "S1", "text": "Evidence"}],
                response_format="json",
            )

    def test_valid_json_is_returned_as_structured_output(self) -> None:
        structured = {"tasks": [], "start_date": None, "unresolved": ["tasks"]}
        runtime = MockRuntime(json.dumps(structured))
        result = LocalTransformerGenerator(
            LocalGenerationConfig(Path("local"), max_new_tokens=8),
            runtime=runtime,
        ).generate(
            "Create a plan",
            [{"source_ref": "S1", "text": "Evidence"}],
            response_format="json",
            task="wbs",
        )
        self.assertEqual(result["structured_output"], structured)

    def test_missing_model_path_is_actionable_and_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing"
            generator = LocalTransformerGenerator(
                LocalGenerationConfig(missing_model, max_new_tokens=8)
            )
            with self.assertRaisesRegex(FileNotFoundError, "Local generative model"):
                generator.generate(
                    "Question?",
                    [{"source_ref": "S1", "text": "Evidence"}],
                )

    def test_empty_evidence_is_rejected(self) -> None:
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(Path("local"), max_new_tokens=8),
            runtime=MockRuntime("answer"),
        )
        with self.assertRaisesRegex(ValueError, "without retrieved evidence"):
            generator.generate("Question?", [])

    def test_int8_quantization_is_cpu_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU only"):
            LocalGenerationConfig(
                Path("local"),
                device="cuda",
                quantization="dynamic-int8",
            )


if __name__ == "__main__":
    unittest.main()

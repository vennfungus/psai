"""Optional inference test for a model already provisioned on the machine."""

import os
from pathlib import Path
import unittest

from generation import LocalGenerationConfig, LocalTransformerGenerator


LOCAL_MODEL_PATH = os.environ.get("PSAI_LOCAL_MODEL_PATH")


@unittest.skipUnless(
    LOCAL_MODEL_PATH,
    "Set PSAI_LOCAL_MODEL_PATH to a pre-provisioned local causal model",
)
class LocalModelIntegrationTests(unittest.TestCase):
    def test_local_model_generates_without_model_hub_access(self) -> None:
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(
                Path(LOCAL_MODEL_PATH),
                device="cpu",
                context_length=2048,
                max_new_tokens=32,
            )
        )
        response = generator.generate(
            "State the evidenced requirement.",
            [
                {
                    "source_ref": "S1",
                    "text": "The project must retain source citations.",
                }
            ],
        )
        self.assertTrue(response["answer"].strip())
        self.assertFalse(response["model_metadata"]["network_access"])


if __name__ == "__main__":
    unittest.main()

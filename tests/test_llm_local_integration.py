"""Optional inference test for an already provisioned local causal model."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from llm import LocalTransformerClient, LocalTransformerConfig

LOCAL_MODEL_PATH = os.environ.get("PSAI_LOCAL_MODEL_PATH")


@unittest.skipUnless(
    LOCAL_MODEL_PATH,
    "Set PSAI_LOCAL_MODEL_PATH to a pre-provisioned local causal model",
)
class LocalTransformerIntegrationTests(unittest.TestCase):
    def test_generation_works_with_hugging_face_offline_flags(self) -> None:
        with patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ):
            client = LocalTransformerClient(
                LocalTransformerConfig(
                    Path(LOCAL_MODEL_PATH),
                    device="auto",
                    context_length=2048,
                    max_new_tokens=32,
                )
            )
            result = client.generate("Write one short greeting.")

        self.assertTrue(result.text.strip())
        self.assertFalse(result.metadata["network_access"])


if __name__ == "__main__":
    unittest.main()

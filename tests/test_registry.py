import unittest
from unittest.mock import patch

from clef_pipeline.registry import create_retriever


class RegistryTests(unittest.TestCase):
    def test_bge_m3_retriever_uses_default_lora_adapter(self):
        with patch("clef_pipeline.registry.BGEM3Retriever") as mock_retriever:
            create_retriever("bge-m3")

        mock_retriever.assert_called_once_with(
            lora_id="boyes-boys-clef-2026/bge-m3-checkthat-finetuned"
        )


if __name__ == "__main__":
    unittest.main()

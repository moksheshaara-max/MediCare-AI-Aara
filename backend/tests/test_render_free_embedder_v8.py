"""Offline smoke tests; do not require downloading the 110MB model."""
import importlib
import os
import sys
import unittest
from unittest.mock import patch


class LowMemoryEmbedderTests(unittest.TestCase):
    def test_import_has_no_torch_side_effect(self):
        from pathlib import Path
        code = (Path(__file__).resolve().parents[1] / 'embeddings' / 'local_embedder.py').read_text(encoding='utf-8-sig')
        self.assertNotIn('import torch\n', code.split('def _get_model():')[0])
        self.assertNotIn('from sentence_transformers import SentenceTransformer\n', code.split('def _get_model():')[0])

    def test_missing_onnx_assets_fails_closed(self):
        from embeddings import local_embedder
        with patch.dict(os.environ, {"MEDICARE_EMBEDDER_BACKEND": "onnx"}):
            with patch.object(local_embedder, '_ONNX_FILE', local_embedder._ROOT / 'missing.onnx'):
                self.assertIsNone(local_embedder.embed_single_text('serious patient care query'))

    def test_empty_string_contract(self):
        from embeddings.local_embedder import embed_single_text
        self.assertEqual(len(embed_single_text('')), 768)


if __name__ == '__main__':
    unittest.main()

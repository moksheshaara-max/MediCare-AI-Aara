"""Pure offline policy checks for optional evaluation on memory-limited runtime."""
import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _read_policy():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_live_rag_evaluation_enabled"]
    if len(found) != 1:
        raise AssertionError("Expected exactly one v11 evaluation policy")
    module = ast.Module(body=found, type_ignores=[])
    namespace = {"os": os}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["_live_rag_evaluation_enabled"]


class LiveEvaluationPolicyTests(unittest.TestCase):
    def test_render_onnx_skips_redundant_evaluation(self):
        policy = _read_policy()
        with patch.dict(os.environ, {"MEDICARE_EMBEDDER_BACKEND": "onnx",
                                     "MEDICARE_LIVE_RAG_EVALUATION": "auto"}):
            self.assertFalse(policy())

    def test_local_torch_keeps_existing_evaluation(self):
        policy = _read_policy()
        with patch.dict(os.environ, {"MEDICARE_EMBEDDER_BACKEND": "torch",
                                     "MEDICARE_LIVE_RAG_EVALUATION": "auto"}):
            self.assertTrue(policy())

    def test_explicit_override(self):
        policy = _read_policy()
        with patch.dict(os.environ, {"MEDICARE_EMBEDDER_BACKEND": "onnx",
                                     "MEDICARE_LIVE_RAG_EVALUATION": "1"}):
            self.assertTrue(policy())
        with patch.dict(os.environ, {"MEDICARE_EMBEDDER_BACKEND": "torch",
                                     "MEDICARE_LIVE_RAG_EVALUATION": "0"}):
            self.assertFalse(policy())

    def test_import_lazy_and_evaluation_output_nullable(self):
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, ast.ImportFrom) and node.module == "evaluator.rag_evaluator"
            for node in tree.body
        ), "Evaluation library must not be imported at Flask startup")
        ask = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ask')
        imports = [n for n in ast.walk(ask) if isinstance(n, ast.ImportFrom)
                   and n.module == 'evaluator.rag_evaluator']
        self.assertEqual(len(imports), 1, 'Keep evaluation available inside optional branch')
        source = APP_PATH.read_text(encoding='utf-8')
        self.assertIn('"evaluation": eval_metrics', source)
        self.assertIn('eval_metrics = None', source)
        self.assertIn('log_memory("ask.after_evaluation")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

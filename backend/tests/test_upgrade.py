"""Offline deterministic unit/regression checks (no MongoDB, PubMed, or Gemini calls).

Run from package root with: python -m unittest discover -s tests -v
These are NOT clinical validation or live end-to-end integration tests.
"""
import importlib.util
import ast
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_pubmed():
    if "xmltodict" not in sys.modules:
        stub = types.ModuleType("xmltodict")
        stub.parse = lambda *_: {}
        sys.modules["xmltodict"] = stub
    spec = importlib.util.spec_from_file_location("pm_local_v3", ROOT / "pubmed" / "pubmed_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_search():
    # PyMongo and its error types are only mocked for OFFLINE unit tests.
    pymongo = types.ModuleType("pymongo")
    errors = types.ModuleType("pymongo.errors")
    class ConnectionFailure(Exception):
        pass
    class AutoReconnect(ConnectionFailure):
        pass
    class NetworkTimeout(ConnectionFailure):
        pass
    class ServerSelectionTimeoutError(ConnectionFailure):
        pass
    errors.ConnectionFailure = ConnectionFailure
    errors.AutoReconnect = AutoReconnect
    errors.NetworkTimeout = NetworkTimeout
    errors.ServerSelectionTimeoutError = ServerSelectionTimeoutError
    pymongo.errors = errors
    class FakeDB:
        def __getitem__(self, key):
            return types.SimpleNamespace(aggregate=lambda q: [], find=lambda *_: [])
    pymongo.MongoClient = lambda *_args, **_kwargs: types.SimpleNamespace(__getitem__=lambda *_: FakeDB())
    # __getitem__ special method should exist on the class
    class FakeMongoClient:
        def __init__(self, *args, **kwargs):
            pass
        def __getitem__(self, name):
            return FakeDB()
    pymongo.MongoClient = FakeMongoClient
    old_p = sys.modules.get("pymongo")
    old_e = sys.modules.get("pymongo.errors")
    sys.modules["pymongo"] = pymongo
    sys.modules["pymongo.errors"] = errors
    try:
        spec = importlib.util.spec_from_file_location("retrieval_v3_local", ROOT / "retrieval" / "search.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, errors
    finally:
        if old_p is None:
            sys.modules.pop("pymongo", None)
        else:
            sys.modules["pymongo"] = old_p
        if old_e is None:
            sys.modules.pop("pymongo.errors", None)
        else:
            sys.modules["pymongo.errors"] = old_e


class PubMedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = load_pubmed()

    def test_hfref_query_remains_about_hfref(self):
        q = "What is the current guideline-based first-line pharmacological treatment for HFrEF?"
        queries = self.pm._query_plan(q)
        self.assertTrue(queries)
        self.assertTrue(all("ejection fraction" in q.lower() or "hfref" in q.lower() for q in queries))
        self.assertTrue(all('"current"' not in q.lower() for q in queries))

    def test_condition_subtype(self):
        p = {"title": "Management of type 1 diabetes mellitus", "abstract": "T1DM", "year": "2024"}
        self.assertFalse(self.pm._is_relevant_paper("treatment for type 2 diabetes", p))
        p["title"] = "Glucose lowering in adults with type 2 diabetes"
        self.assertTrue(self.pm._is_relevant_paper("treatment for type 2 diabetes", p))

    def test_no_unanchored_fallback(self):
        self.assertEqual([], self.pm._query_plan("Current guideline-based first-line treatment?"))

    def test_question_topic_order(self):
        first = self.pm._topics("Explain chronic kidney disease and type 2 diabetes overlap")[0]
        self.assertEqual("chronic kidney disease", first)
        first = self.pm._topics("Explain type 2 diabetes with chronic kidney disease")[0]
        self.assertEqual("type 2 diabetes", first)
        self.assertEqual("heart failure with reduced ejection fraction", self.pm._topics("heart failure with reduced ejection fraction") [0])

    def test_multi_drug_comparison_retained(self):
        terms = self.pm._query_plan("Compare tirzepatide versus semaglutide for NASH")
        self.assertIn("tirzepatide", terms[0].lower())
        self.assertIn("semaglutide", terms[0].lower())
        self.assertIn("steatohepatitis", terms[0].lower())

    def test_pediatric_papers_not_adult_only(self):
        paper = {"title": "Type 1 diabetes therapy in adults", "abstract": "Adults with type 1 diabetes"}
        self.assertFalse(self.pm._is_relevant_paper("type 1 diabetes in a 10-year-old child", paper))
        paper["title"] = "Type 1 diabetes therapy in children"
        self.assertTrue(self.pm._is_relevant_paper("type 1 diabetes in a 10-year-old child", paper))

    def test_relevance_filter_does_not_accept_generic_paper(self):
        paper = {"title": "Management of hypertension", "abstract": "Blood pressure therapy"}
        self.assertFalse(self.pm._is_relevant_paper("first-line HFrEF treatment", paper))

    def test_fetch_fails_open(self):
        with patch.object(self.pm, "_request", side_effect=self.pm.requests.exceptions.ConnectionError("mocked offline")):
            self.assertEqual(self.pm.fetch_pubmed_papers(["123"]), [])

    def test_search_stays_anchored_after_empty_ids(self):
        seen = []
        class Empty:
            def json(self):
                return {"esearchresult": {"idlist": []}}
        def fake(url, params):
            seen.append(params["term"])
            return Empty()
        with patch.object(self.pm, "_request", side_effect=fake):
            self.assertEqual(self.pm.search_pubmed_ids("current management of HFrEF"), [])
        self.assertTrue(seen and all("ejection fraction" in q.lower() for q in seen))


class GeneratorBudgetTests(unittest.TestCase):
    def test_mode_adaptive_pubmed_budget(self):
        # Execute only the pure budget helper to avoid importing Gemini clients.
        s, _ = load_search()
        filename = ROOT / "generation" / "generator.py"
        tree = ast.parse(filename.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pubmed_budget")
        code = compile(ast.Module(body=[fn], type_ignores=[]), str(filename), "exec")
        g = {"normalize_assistant_mode": lambda m: "pg_student" if m == "pg_student" else "doctor"}
        old_module = sys.modules.get("retrieval.search")
        old_pkg = sys.modules.get("retrieval")
        sys.modules["retrieval"] = types.ModuleType("retrieval")
        sys.modules["retrieval.search"] = types.SimpleNamespace(classify_pg_query_depth=s.classify_pg_query_depth)
        try:
            exec(code, g)
            fn = g["_pubmed_budget"]
            self.assertEqual(5, fn("Explain CKD comprehensively and in detail", "doctor")[0])
            self.assertEqual(5, fn("Explain CKD", "pg_student")[0])
            self.assertEqual(7, fn("Explain CKD in detail", "pg_student")[0])
            self.assertEqual(9, fn("Teach CKD comprehensively: mechanism, diagnosis, management, complications and pharmacology", "pg_student")[0])
        finally:
            if old_module is None:
                sys.modules.pop("retrieval.search", None)
            else:
                sys.modules["retrieval.search"] = old_module
            if old_pkg is None:
                sys.modules.pop("retrieval", None)
            else:
                sys.modules["retrieval"] = old_pkg


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.errors = load_search()

    def test_depth_budgets(self):
        s = self.s
        self.assertEqual(36, s.pg_retrieval_budget("What is hypertension?"))
        self.assertEqual(45, s.pg_retrieval_budget("Explain hypertension in detail, including treatment and mechanism"))
        self.assertEqual(50, s.pg_retrieval_budget("Explain CKD comprehensively: pathophysiology, staging, complications, pharmacology, diagnosis, treatment and monitoring"))

    def test_basic_query_stays_simple(self):
        s = self.s
        self.assertEqual("simple", s.classify_pg_query_depth("What is nephrotic syndrome? Explain its basic pathophysiology and major clinical features."))
        self.assertEqual("simple", s.classify_pg_query_depth("Explain the mechanism of action, major adverse effects and important contraindications of ACE inhibitors."))

    def test_no_t1dm_t2dm_guideline_leakage(self):
        s = self.s
        t1 = {"official_title": "Type 1 diabetes management", "topic": "type 1 diabetes", "specialty": "endocrinology"}
        self.assertTrue(s._has_explicit_disease_context_conflict("Treat type 2 diabetes", t1))
        self.assertFalse(s._has_explicit_disease_context_conflict("Treat type 1 diabetes", t1))

    def test_no_hypertension_guideline_for_hfref(self):
        s = self.s
        meta = {"official_title":"Hypertension clinical guideline", "topic":"hypertension"}
        self.assertTrue(s._has_explicit_disease_context_conflict("HFrEF pharmacological treatment", meta))

    def test_mongodb_transient_retry(self):
        s = self.s
        class FakeCollection:
            calls = 0
            def aggregate(self, pipeline):
                self.calls += 1
                if self.calls == 1:
                    raise self_err.AutoReconnect("fake temporary disconnect")
                return [{"source_id":"x", "text":"valid"}]
        self_err = self.errors
        c = FakeCollection()
        with patch.object(s, "collection", c), patch.object(s.time, "sleep"):
            rows = s._vector_search_candidates([0.0]*768, 10, 50)
        self.assertEqual(c.calls, 2)
        self.assertEqual(len(rows), 1)

    def test_non_transient_failure_not_hidden(self):
        s = self.s
        with patch.object(s, "collection", types.SimpleNamespace(aggregate=lambda x: (_ for _ in ()).throw(ValueError("invalid index")))):
            with self.assertRaisesRegex(ValueError, "invalid index"):
                s._vector_search_candidates([0.0]*768, 10, 50)

    def test_dated_comparison_pair(self):
        s = self.s
        def chunk(sid, year, page):
            return {"source_id":sid, "source_type":"clinical_guideline", "official_title":f"{year} Hypertension Guidelines", "topic":"hypertension", "publication_year":year, "specialty":"cardiology", "country":"India", "vector_score":0.86-(2026-year)*0.01, "retrieval_score":0.97, "score":0.87, "page":page, "chunk_id":f"{sid}-{page}", "text":f"Hypertension evidence {year} page {page}: differentiated recommendations and methods in adults."}
        pool = [chunk("old_2024",2024,1),chunk("old_2024",2024,2),chunk("new_2026",2026,1),chunk("new_2026",2026,2)]
        extra, source_years = s._comparison_guideline_pairs("Compare older and current hypertension guidelines for adults", pool)
        self.assertEqual({2024,2026},{y for _,y in source_years})
        final = s._ensure_comparison_pair([],extra,36)
        self.assertGreaterEqual(len({x['source_id'] for x in final}),2)


if __name__ == '__main__':
    unittest.main()

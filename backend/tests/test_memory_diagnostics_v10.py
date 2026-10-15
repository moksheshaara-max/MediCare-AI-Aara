"""No-network, standard-library checks for optional Render memory diagnostics."""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_diagnostics as diag


class MemoryDiagnosticsTests(unittest.TestCase):
    def test_disabled_is_silent_and_does_not_change_return(self):
        @diag.trace_request("test")
        def api_handler():
            diag.log_memory("test.internal")
            return {"status": 200}

        output = io.StringIO()
        with patch.dict(os.environ, {"MEDICARE_MEMORY_LOGGING": "0"}):
            with contextlib.redirect_stdout(output):
                self.assertEqual(api_handler(), {"status": 200})
        self.assertEqual(output.getvalue(), "")

    def test_enabled_logs_stages_without_patient_data(self):
        @diag.trace_request("test")
        def api_handler():
            diag.log_memory("test.internal")
            return {"status": 200}

        output = io.StringIO()
        with patch.dict(os.environ, {"MEDICARE_MEMORY_LOGGING": "1"}):
            with contextlib.redirect_stdout(output):
                self.assertEqual(api_handler(), {"status": 200})
        logged = output.getvalue()
        for stage in ("test.start", "test.internal", "test.finish"):
            self.assertIn("[MEM] stage=" + stage, logged)
        self.assertIn("rss_mb=", logged)
        self.assertIn("cgroup_mb=", logged)
        self.assertIn("elapsed_s=", logged)

    def test_exception_preserved_and_sampler_counter_restored(self):
        @diag.trace_request("test_failure")
        def failing_api_handler():
            raise ValueError("expected test failure")

        output = io.StringIO()
        with patch.dict(os.environ, {"MEDICARE_MEMORY_LOGGING": "1"}):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(ValueError, "expected test failure"):
                    failing_api_handler()
        self.assertIn("[MEM] stage=test_failure.finish", output.getvalue())
        self.assertEqual(diag._active_requests, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

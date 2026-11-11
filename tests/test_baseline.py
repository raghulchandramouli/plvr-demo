import json
from pathlib import Path
import tempfile
import unittest

from plvr3b.baseline import audit_run, run_baseline
from plvr3b.baseline_tasks import build_smoke_cases, public_input


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self, root, config):
        self.config = config
        self.metadata = {"model_load_count": 1, "backend": "test-double"}
        self.last_observations = {}
        self.examples = {json.dumps(public_input(case), sort_keys=True): case for case in build_smoke_cases()}
        self.case = None

    def peak_memory(self):
        return 0

    def prepare(self, public):
        self.case = self.examples[json.dumps(public, sort_keys=True)]
        return json.dumps(public), [1, 2, 3]

    def generate(self, tokens, deadline):
        yield {"text": json.dumps(self.case.oracle), "token_id": 7, "prompt_tokens": 3, "generation_tokens": 1, "finish_reason": "stop"}


class FailingModel(FakeModel):
    def generate(self, tokens, deadline):
        yield {"text": "{", "token_id": 7, "prompt_tokens": 3, "generation_tokens": 1, "finish_reason": None}
        if self.case.id == "smoke-001":
            raise TimeoutError("test deadline")
        raise RuntimeError("test generation failure")


class MemoryFailureModel(FakeModel):
    def generate(self, tokens, deadline):
        yield {"text": "{", "token_id": 7, "prompt_tokens": 3, "generation_tokens": 1, "finish_reason": None}
        raise MemoryError("test memory budget")


class PrepareFailureModel(FakeModel):
    def prepare(self, public):
        raise ValueError("test prompt budget rejection")


class BaselineArtifactTests(unittest.TestCase):
    def test_one_backend_load_and_all_thirty_calls_reconcile(self):
        loads = []

        def factory(root, config):
            loads.append(1)
            return FakeModel(root, config)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_baseline(ROOT, output, backend_factory=factory)
            self.assertEqual(loads, [1])
            self.assertEqual(summary["outcomes"], {"success": 30})
            self.assertEqual(summary["model_calls"], 30)
            self.assertEqual(summary["prompt_tokens"], 90)
            self.assertEqual(summary["generation_tokens"], 30)
            self.assertEqual(summary["model_load_count"], 1)
            self.assertGreater(summary["wall_seconds"], summary["model_load_seconds"])
            self.assertEqual(audit_run(output)["audit"], "passed")
            with self.assertRaises(FileExistsError):
                run_baseline(ROOT, output, backend_factory=factory)

    def test_timeout_and_runtime_failure_keep_raw_output_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_baseline(ROOT, output, limit=2, backend_factory=FailingModel)
            self.assertEqual(summary["outcomes"], {"runtime_failure": 1, "timeout": 1})
            self.assertEqual(summary["generation_tokens"], 2)
            self.assertEqual(summary["model_calls"], 2)
            trace = json.loads((output / "traces/smoke-001.json").read_text())
            self.assertEqual(trace["raw_output"], "{")
            self.assertEqual(trace["token_ids"], [7])
            self.assertTrue(trace["errors"])
            self.assertEqual(trace["evaluation"]["status"], "format_failure")
            self.assertEqual(audit_run(output)["audit"], "passed")

    def test_resource_abort_keeps_remaining_cases_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_baseline(ROOT, output, limit=3, backend_factory=MemoryFailureModel)
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["outcomes"], {"not_run": 2, "runtime_failure": 1})
            self.assertEqual(summary["planned_examples"], 3)
            self.assertEqual(summary["model_calls"], 1)
            self.assertEqual(audit_run(output)["audit"], "passed")

    def test_oracle_only_never_loads_model_and_can_be_replayed(self):
        def forbidden(*args):
            self.fail("Oracle validation loaded a model")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_baseline(ROOT, output, oracle_only=True, backend_factory=forbidden)
            self.assertEqual(summary["model_load_count"], 0)
            self.assertEqual(summary["model_calls"], 0)
            self.assertEqual(summary["outcomes"], {"oracle_validated": 30})
            self.assertEqual(audit_run(output)["audit"], "passed")

    def test_tampered_counts_and_raw_output_fail_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_baseline(ROOT, output, limit=1, backend_factory=FakeModel)
            path = output / "summary.json"
            summary = json.loads(path.read_text())
            summary["generation_tokens"] += 1
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "generation_tokens"):
                audit_run(output)

    def test_prepare_rejection_records_zero_calls_and_is_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_baseline(ROOT, output, limit=1, backend_factory=PrepareFailureModel)
            self.assertEqual(summary["model_calls"], 0)
            self.assertEqual(summary["outcomes"], {"runtime_failure": 1})
            self.assertEqual(audit_run(output)["audit"], "passed")

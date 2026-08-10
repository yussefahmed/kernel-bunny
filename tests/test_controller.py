from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.llm import ProviderError
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.worker import Worker


def proposal(source: str, intent: str = "test intent") -> str:
    return json.dumps({"worker_intent": intent, "cuda_source": source})


class ScriptedWorker:
    def __init__(self, initial: str | Exception, repairs: list[str | Exception] | None = None):
        self.initial = initial
        self.repairs = list(repairs or [])
        self.repair_requests: list[dict[str, object]] = []

    def request_proposal(
        self,
        parent: Candidate,
        parent_evaluation: Evaluation,
        global_best_runtime_ms: float,
        **kwargs,
    ) -> str:
        if isinstance(self.initial, Exception):
            raise self.initial
        return self.initial

    def request_repair(
        self,
        original_parent: Candidate,
        failed_candidate: Candidate,
        failure_kind: str,
        diagnostic: str,
    ) -> str:
        self.repair_requests.append(
            {
                "original_parent": original_parent,
                "failed_candidate": failed_candidate,
                "failure_kind": failure_kind,
                "diagnostic": diagnostic,
            }
        )
        response = self.repairs.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @staticmethod
    def parse_proposal(raw_response: str):
        return Worker.parse_proposal(raw_response)


class SourceEvaluator:
    def __init__(self):
        self.candidates: list[Candidate] = []

    def evaluate(self, candidate: Candidate) -> Evaluation:
        self.candidates.append(candidate)
        source = Path(candidate.source_path).read_text(encoding="utf-8")
        if source == "compile_bad":
            return Evaluation(
                candidate.revision_id,
                False,
                "compiler says syntax error",
                None,
                "not run",
                None,
                "not run",
            )
        if source == "wrong":
            return Evaluation(
                candidate.revision_id,
                True,
                "compiled",
                False,
                "Failed M=17, N=19, K=13, seed=37",
                None,
                "not run",
            )
        if source == "benchmark_bad":
            return Evaluation(
                candidate.revision_id,
                True,
                "compiled",
                True,
                "correct",
                None,
                "timing failed",
            )
        runtime = 20.0 if source == "slow" else 0.5
        return Evaluation(
            candidate.revision_id,
            True,
            "compiled",
            True,
            "correct",
            runtime,
            "benchmarked",
        )


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        parent_path = self.root / "baseline.cu"
        parent_path.write_text("good", encoding="utf-8")
        self.parent = Candidate("baseline", None, str(parent_path), revision=0)
        self.parent_evaluation = Evaluation(
            "baseline/r0", True, "compiled", True, "correct", 1.0, "benchmarked"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_controller(
        self,
        worker: ScriptedWorker,
        evaluator: SourceEvaluator | None = None,
        config: ControllerConfig | None = None,
    ):
        evaluator = evaluator or SourceEvaluator()
        controller = Controller(worker, evaluator, self.root / "runs", config)
        return controller.run_candidate(self.parent, self.parent_evaluation), evaluator

    def test_compile_then_correctness_repair_reaches_r2(self) -> None:
        worker = ScriptedWorker(
            proposal("compile_bad"),
            [proposal("wrong"), proposal("good")],
        )
        result, evaluator = self.run_controller(worker)

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.final_revision_id, "C0001/r2")
        self.assertEqual(result.compile_repairs_used, 1)
        self.assertEqual(result.correctness_repairs_used, 1)
        self.assertEqual(
            [candidate.parent_id for candidate in evaluator.candidates],
            ["baseline/r0", "C0001/r0", "C0001/r1"],
        )
        self.assertTrue((self.root / "runs/C0001/r2/evaluation.json").is_file())

    def test_compile_budget_exhaustion(self) -> None:
        worker = ScriptedWorker(proposal("compile_bad"), [proposal("compile_bad")])
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "compile_repair_exhausted")
        self.assertEqual(result.final_revision_id, "C0001/r1")
        self.assertEqual(result.compile_repairs_used, 1)
        self.assertEqual(len(worker.repair_requests), 1)
        self.assertEqual(
            worker.repair_requests[0]["diagnostic"],
            "compiler says syntax error",
        )

    def test_correctness_budget_exhaustion(self) -> None:
        worker = ScriptedWorker(proposal("wrong"), [proposal("wrong")])
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "correctness_repair_exhausted")
        self.assertEqual(result.final_revision_id, "C0001/r1")
        self.assertEqual(result.correctness_repairs_used, 1)
        self.assertEqual(len(worker.repair_requests), 1)

    def test_correctness_diagnostic_is_sanitized(self) -> None:
        worker = ScriptedWorker(proposal("wrong"), [proposal("good")])
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "evaluated")
        diagnostic = str(worker.repair_requests[0]["diagnostic"])
        self.assertIn("held-out correctness case", diagnostic)
        self.assertNotIn("M=17", diagnostic)
        self.assertEqual(worker.repair_requests[0]["failure_kind"], "correctness")

    def test_correctness_repair_can_use_compile_budget(self) -> None:
        worker = ScriptedWorker(
            proposal("wrong"),
            [proposal("compile_bad"), proposal("good")],
        )
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.final_revision_id, "C0001/r2")
        self.assertEqual(result.correctness_repairs_used, 1)
        self.assertEqual(result.compile_repairs_used, 1)

    def test_valid_slow_candidate_is_not_repaired(self) -> None:
        worker = ScriptedWorker(proposal("slow"))
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.evaluation.runtime_ms, 20.0)
        self.assertEqual(worker.repair_requests, [])

    def test_initial_provider_failure_does_not_consume_budget(self) -> None:
        worker = ScriptedWorker(ProviderError("offline"))
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "provider_failed")
        self.assertFalse(result.budget_consumed)
        self.assertIsNone(result.final_revision_id)

    def test_malformed_initial_response_consumes_budget(self) -> None:
        worker = ScriptedWorker("not json")
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "proposal_failed")
        self.assertTrue(result.budget_consumed)

    def test_repair_provider_failure_is_terminal_and_counted(self) -> None:
        worker = ScriptedWorker(
            proposal("compile_bad"),
            [ProviderError("offline")],
        )
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "repair_provider_failed")
        self.assertTrue(result.budget_consumed)
        self.assertEqual(result.compile_repairs_used, 1)
        self.assertEqual(result.final_revision_id, "C0001/r0")

    def test_malformed_repair_is_terminal_and_counted(self) -> None:
        worker = ScriptedWorker(proposal("compile_bad"), ["not json"])
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "repair_proposal_failed")
        self.assertEqual(result.compile_repairs_used, 1)
        self.assertEqual(result.final_revision_id, "C0001/r0")
        self.assertTrue((self.root / "runs/C0001/r1/raw_response.txt").is_file())

    def test_benchmark_failure_is_not_sent_for_repair(self) -> None:
        worker = ScriptedWorker(proposal("benchmark_bad"))
        result, _ = self.run_controller(worker)

        self.assertEqual(result.status, "benchmark_failed")
        self.assertEqual(worker.repair_requests, [])


if __name__ == "__main__":
    unittest.main()

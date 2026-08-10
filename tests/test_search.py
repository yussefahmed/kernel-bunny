from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernelbunny.controller import (
    Controller,
    ControllerConfig,
    PopulationMember,
    _select_population,
)
from kernelbunny.llm import ProviderError
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.worker import Worker


def proposal(source: str) -> str:
    return json.dumps(
        {
            "worker_intent": f"Return the scripted outcome {source}.",
            "cuda_source": source,
        }
    )


class SequenceWorker:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, float]] = []

    def request_proposal(
        self,
        parent: Candidate,
        parent_evaluation: Evaluation,
        global_best_runtime_ms: float,
        **kwargs,
    ) -> str:
        self.requests.append((parent.revision_id, global_best_runtime_ms))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def request_repair(self, *args, **kwargs) -> str:
        raise AssertionError("Search tests disable candidate repair budgets.")

    @staticmethod
    def parse_proposal(raw_response: str):
        return Worker.parse_proposal(raw_response)


class SearchEvaluator:
    def evaluate(self, candidate: Candidate) -> Evaluation:
        source = Path(candidate.source_path).read_text(encoding="utf-8")
        if source == "compile_bad":
            return Evaluation(
                candidate.revision_id,
                False,
                "syntax error",
                None,
                "not run",
                None,
                "not run",
            )
        runtime = float(source.removeprefix("runtime:"))
        return Evaluation(
            candidate.revision_id,
            True,
            "compiled",
            True,
            "correct",
            runtime,
            "benchmarked",
        )


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        baseline_path = self.root / "baseline.cu"
        baseline_path.write_text("runtime:1.0", encoding="utf-8")
        self.baseline = Candidate("baseline", None, str(baseline_path))
        self.baseline_evaluation = Evaluation(
            "baseline/r0",
            True,
            "compiled",
            True,
            "correct",
            1.0,
            "benchmarked",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def config(self, **overrides) -> ControllerConfig:
        values = {
            "max_compile_repairs": 0,
            "max_correctness_repairs": 0,
            "population_size": 3,
            "children_per_parent": 2,
            "max_generations": 6,
            "max_candidates": 12,
            "max_no_improvement_generations": 6,
            "meaningful_improvement_ratio": 0.02,
        }
        values.update(overrides)
        return ControllerConfig(**values)

    def run_search(self, responses, **config_overrides):
        worker = SequenceWorker(responses)
        controller = Controller(
            worker,
            SearchEvaluator(),
            self.root / "runs",
            self.config(**config_overrides),
        )
        result = controller.run_search(self.baseline, self.baseline_evaluation)
        return result, worker

    def member(self, name: str, runtime: float) -> PopulationMember:
        path = self.root / f"{name}.cu"
        path.write_text(f"runtime:{runtime}", encoding="utf-8")
        candidate = Candidate(name, None, str(path))
        evaluation = Evaluation(
            candidate.revision_id,
            True,
            "compiled",
            True,
            "correct",
            runtime,
            "benchmarked",
        )
        return PopulationMember(candidate, evaluation)

    def test_selection_keeps_two_fastest_and_unrepresented_lineage(self) -> None:
        entries = [
            (self.member("P1", 0.300), "P1/r0"),
            (self.member("A", 0.291), "P1/r0"),
            (self.member("P2", 0.326), "P2/r0"),
            (self.member("C", 0.294), "P2/r0"),
            (self.member("D", 0.410), "P2/r0"),
            (self.member("P3", 0.355), "P3/r0"),
            (self.member("E", 0.303), "P3/r0"),
        ]

        selected = _select_population(entries, 3)

        self.assertEqual(
            [member.candidate.revision_id for member in selected],
            ["A/r0", "C/r0", "E/r0"],
        )

    def test_initial_parent_expands_to_population_of_three(self) -> None:
        result, _ = self.run_search(
            [proposal("runtime:0.90"), proposal("runtime:0.95")],
            max_generations=1,
            max_candidates=2,
        )

        self.assertEqual(
            [member.evaluation.runtime_ms for member in result.active_population],
            [0.90, 0.95, 1.0],
        )
        self.assertEqual(len(result.steps[0].attempts), 2)

    def test_all_failed_children_preserve_parent_population(self) -> None:
        result, _ = self.run_search(
            [proposal("compile_bad"), proposal("compile_bad")],
            max_generations=1,
            max_candidates=2,
        )

        self.assertEqual(
            [member.candidate.revision_id for member in result.active_population],
            ["baseline/r0"],
        )
        self.assertEqual(result.best_candidate.revision_id, "baseline/r0")
        self.assertEqual(result.no_improvement_generations, 1)

    def test_small_gain_updates_best_without_being_meaningful(self) -> None:
        result, _ = self.run_search(
            [proposal("runtime:0.99"), proposal("runtime:1.20")],
            max_generations=1,
            max_candidates=2,
        )

        self.assertEqual(result.best_candidate.revision_id, "C0001/r0")
        self.assertFalse(result.steps[0].meaningful_improvement)
        self.assertEqual(result.no_improvement_generations, 1)
        self.assertIn(
            "C0001/r0",
            [member.candidate.revision_id for member in result.active_population],
        )

    def test_progress_on_slower_branch_resets_stagnation(self) -> None:
        result, _ = self.run_search(
            [
                proposal("runtime:0.29"),
                proposal("runtime:0.35"),
                proposal("runtime:0.40"),
                proposal("runtime:0.34"),
                proposal("compile_bad"),
            ],
            max_generations=2,
            max_candidates=5,
        )

        self.assertEqual(result.best_evaluation.runtime_ms, 0.29)
        self.assertTrue(result.steps[1].meaningful_improvement)
        self.assertEqual(result.no_improvement_generations, 0)
        self.assertEqual(
            [attempt.parent_revision_id for attempt in result.steps[1].attempts],
            ["C0001/r0", "C0002/r0", "baseline/r0"],
        )

    def test_no_improvement_counts_once_per_generation(self) -> None:
        result, _ = self.run_search(
            [
                proposal("compile_bad"),
                proposal("compile_bad"),
                proposal("compile_bad"),
                proposal("compile_bad"),
            ],
            max_generations=2,
            max_candidates=4,
        )

        self.assertEqual(result.generations_attempted, 2)
        self.assertEqual(result.candidates_consumed, 4)
        self.assertEqual(result.no_improvement_generations, 2)

    def test_provider_failures_advance_generation_without_other_budgets(self) -> None:
        result, _ = self.run_search(
            [ProviderError("offline"), ProviderError("offline")],
            max_generations=1,
            max_candidates=2,
        )

        self.assertEqual(result.stop_reason, "max_generations")
        self.assertEqual(result.generations_attempted, 1)
        self.assertEqual(result.candidates_consumed, 0)
        self.assertEqual(result.no_improvement_generations, 0)

    def test_partial_generation_schedules_parents_round_robin(self) -> None:
        result, worker = self.run_search(
            [
                proposal("runtime:0.80"),
                proposal("runtime:0.90"),
                proposal("runtime:0.79"),
                proposal("runtime:0.89"),
                proposal("runtime:0.99"),
            ],
            max_generations=3,
            max_candidates=5,
        )

        self.assertEqual(result.stop_reason, "max_candidates")
        self.assertEqual(
            [parent for parent, _ in worker.requests],
            [
                "baseline/r0",
                "baseline/r0",
                "C0001/r0",
                "C0002/r0",
                "baseline/r0",
            ],
        )
        self.assertEqual(
            [attempt.child_number for attempt in result.steps[1].attempts],
            [1, 1, 1],
        )

    def test_selection_does_not_inject_an_external_best(self) -> None:
        archived_best = self.member("archived", 0.10)
        entries = [
            (self.member("P1", 0.20), "P1/r0"),
            (self.member("P2", 0.30), "P2/r0"),
            (self.member("P3", 0.40), "P3/r0"),
        ]

        selected = _select_population(entries, 3)

        self.assertNotIn(archived_best, selected)
        self.assertEqual(len(selected), 3)

    def test_search_state_persists_population_decisions(self) -> None:
        result, _ = self.run_search(
            [proposal("runtime:0.90"), proposal("runtime:1.10")],
            max_generations=1,
            max_candidates=2,
        )
        state_path = self.root / "runs" / "searches" / result.search_id / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], "max_candidates")
        self.assertEqual(
            [entry["revision_id"] for entry in state["active_population"]],
            list(result.steps[0].selected_revision_ids),
        )
        self.assertEqual(state["best_ever"]["revision_id"], "C0001/r0")
        self.assertEqual(len(state["generations"][0]["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()

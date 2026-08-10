from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.curator import (
    CurationEvidence,
    Curator,
    CuratorError,
    CuratorProposal,
)
from kernelbunny.llm import ProviderError
from kernelbunny.memory import MemoryBank
from kernelbunny.models import Candidate, Evaluation, MemoryLesson, Reflection
from kernelbunny.reflector import Reflector
from kernelbunny.worker import Worker


def reflection(candidate_id: str, confidence: str = "medium") -> Reflection:
    return Reflection(
        candidate_id,
        "Changed memory access.",
        "The trusted harness recorded the outcome.",
        "Coalescing may explain the result.",
        confidence,
    )


def evidence(
    candidate_id: str,
    runtime: float | None,
    *,
    parent_runtime: float = 1.0,
) -> CurationEvidence:
    valid = runtime is not None
    return CurationEvidence(
        candidate_id=candidate_id,
        parent_id="baseline/r0",
        terminal_status="evaluated" if valid else "compile_repair_exhausted",
        compile_success=valid,
        correctness_success=True if valid else None,
        runtime_ms=runtime,
        parent_runtime_ms=parent_runtime,
        reflection=reflection(candidate_id),
    )


def curator_response(
    candidate_ids: list[str],
    existing_lesson_id: str | None = None,
) -> str:
    return json.dumps(
        {
            "lessons": [
                {
                    "existing_lesson_id": existing_lesson_id,
                    "lesson": "Prefer coalesced global-memory access when validated.",
                    "evidence_candidate_ids": candidate_ids,
                }
            ]
        }
    )


class RecordingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class CuratorContractTests(unittest.TestCase):
    def test_prompt_contains_sanitized_evidence_without_source_or_hidden_inputs(self) -> None:
        llm = RecordingLLM('{"lessons": []}')
        curator = Curator(llm)
        item = CurationEvidence(
            "C0001/r0",
            "baseline/r0",
            "evaluated",
            True,
            True,
            0.8,
            1.0,
            Reflection(
                "C0001/r0",
                "Introduced tiling.",
                "Correctness passed and runtime improved.",
                "Reuse may have reduced traffic.",
                "medium",
            ),
        )

        curator.request_curation((item,), ())

        prompt = llm.prompts[0]
        self.assertIn("trusted runtime improvement=20.00%", prompt)
        self.assertIn("Post-hoc hypothesis", prompt)
        self.assertNotIn("M=", prompt)
        self.assertNotIn("seed=", prompt)
        self.assertNotIn("candidate.cu", prompt)

    def test_parser_rejects_model_controlled_or_unknown_references(self) -> None:
        payload = json.loads(curator_response(["C0001/r0"]))
        payload["lessons"][0]["status"] = "active"
        with self.assertRaises(CuratorError):
            Curator.parse_proposals(
                json.dumps(payload),
                allowed_evidence_ids={"C0001/r0"},
                allowed_existing_lesson_ids=set(),
                max_lessons=5,
            )

        with self.assertRaises(CuratorError):
            Curator.parse_proposals(
                curator_response(["C9999/r0"]),
                allowed_evidence_ids={"C0001/r0"},
                allowed_existing_lesson_ids=set(),
                max_lessons=5,
            )

    def test_parser_accepts_known_existing_lesson(self) -> None:
        proposals = Curator.parse_proposals(
            curator_response(["C0002/r0"], "L0001"),
            allowed_evidence_ids={"C0002/r0"},
            allowed_existing_lesson_ids={"L0001"},
            max_lessons=5,
        )

        self.assertEqual(proposals[0].existing_lesson_id, "L0001")


class MemoryBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def apply(self, bank, proposals, items, number):
        return bank.apply(
            tuple(proposals),
            tuple(items),
            curation_id=f"S0001/K{number:04d}",
            large_improvement_ratio=0.10,
        )

    def test_large_improvement_activates_single_evidence_lesson(self) -> None:
        bank = MemoryBank(self.root, 10)
        update = self.apply(
            bank,
            [CuratorProposal(None, "Use tiling.", ("C0001/r0",))],
            [evidence("C0001/r0", 0.8)],
            1,
        )

        self.assertEqual(update.accepted_lesson_ids, ("L0001",))
        self.assertEqual(bank.active_lessons()[0].status, "active")
        self.assertTrue((self.root / "global_memory.json").is_file())
        self.assertTrue((self.root / "global_memory.md").is_file())
        self.assertTrue((self.root / "curation_archive.jsonl").is_file())

    def test_repeated_evidence_promotes_low_priority_lesson(self) -> None:
        bank = MemoryBank(self.root, 10)
        self.apply(
            bank,
            [CuratorProposal(None, "Try coalescing.", ("C0001/r0",))],
            [evidence("C0001/r0", 0.95)],
            1,
        )
        self.assertEqual(bank.curator_context()[0].status, "low_priority")

        self.apply(
            bank,
            [CuratorProposal("L0001", "Try coalescing.", ("C0002/r0",))],
            [evidence("C0002/r0", 0.96)],
            2,
        )

        lesson = bank.active_lessons()[0]
        self.assertEqual(lesson.evidence_candidate_ids, ("C0001/r0", "C0002/r0"))

    def test_unreinforced_low_priority_lesson_is_archived(self) -> None:
        bank = MemoryBank(self.root, 10)
        self.apply(
            bank,
            [CuratorProposal(None, "Tentative idea.", ("C0001/r0",))],
            [evidence("C0001/r0", 0.95)],
            1,
        )
        self.apply(bank, [], [evidence("C0002/r0", 0.95)], 2)

        payload = json.loads(
            (self.root / "global_memory.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["lessons"][0]["status"], "archived")

    def test_active_cap_preserves_strongest_evidence(self) -> None:
        bank = MemoryBank(self.root, 1)
        items = [evidence("C0001/r0", 0.8), evidence("C0002/r0", 0.8)]
        self.apply(
            bank,
            [
                CuratorProposal(None, "One support.", ("C0001/r0",)),
                CuratorProposal(None, "Two supports.", ("C0001/r0", "C0002/r0")),
            ],
            items,
            1,
        )

        self.assertEqual(bank.active_lessons()[0].text, "Two supports.")


class CapturingWorker:
    def __init__(self, sources: list[str]) -> None:
        self.sources = list(sources)
        self.contexts: list[dict[str, object]] = []

    def request_proposal(self, *args, **kwargs) -> str:
        self.contexts.append(kwargs)
        source = self.sources.pop(0)
        return json.dumps({"worker_intent": "Test search context.", "cuda_source": source})

    def request_repair(self, *args, **kwargs) -> str:
        raise AssertionError("Repairs are disabled in this test.")

    parse_proposal = staticmethod(Worker.parse_proposal)


class RuntimeEvaluator:
    def evaluate(self, candidate: Candidate) -> Evaluation:
        runtime = float(Path(candidate.source_path).read_text().removeprefix("runtime:"))
        return Evaluation(candidate.revision_id, True, "compiled", True, "correct", runtime, "timed")


class StaticReflector:
    def request_reflection(self, **kwargs) -> str:
        return json.dumps(
            {
                "what_changed": "Changed the test runtime.",
                "observed": "Compilation and correctness passed.",
                "hypothesis": "The test change may be faster.",
                "confidence": "medium",
            }
        )

    parse_reflection = staticmethod(Reflector.parse_reflection)


class ScriptedCurator:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.evidence_sizes: list[int] = []

    def request_curation(self, evidence, existing_lessons, max_lessons=5) -> str:
        self.evidence_sizes.append(len(evidence))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    parse_proposals = staticmethod(Curator.parse_proposals)


class ControllerCurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        baseline_path = self.root / "baseline.cu"
        baseline_path.write_text("runtime:1.0", encoding="utf-8")
        self.baseline = Candidate("baseline", None, str(baseline_path))
        self.baseline_evaluation = Evaluation(
            "baseline/r0", True, "compiled", True, "correct", 1.0, "timed"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def controller(self, worker, curator, *, generations=2):
        return Controller(
            worker,
            RuntimeEvaluator(),
            self.root / "runs",
            ControllerConfig(
                max_compile_repairs=0,
                max_correctness_repairs=0,
                population_size=1,
                children_per_parent=1,
                max_generations=generations,
                max_candidates=generations,
                max_no_improvement_generations=generations,
                curation_interval_generations=1,
            ),
            reflector=StaticReflector(),
            curator=curator,
            memory_dir=self.root / "memory",
        )

    def test_curation_updates_next_worker_context_and_branch_history(self) -> None:
        worker = CapturingWorker(["runtime:0.8", "runtime:0.7"])
        curator = ScriptedCurator(
            [curator_response(["C0001/r0"]), '{"lessons": []}']
        )

        result = self.controller(worker, curator).run_search(
            self.baseline, self.baseline_evaluation
        )

        self.assertEqual([step.status for step in result.curations], ["completed", "completed"])
        self.assertEqual(worker.contexts[0]["global_memory"], ())
        second_memory = worker.contexts[1]["global_memory"]
        self.assertEqual(second_memory[0].id, "L0001")
        second_history = worker.contexts[1]["branch_history"]
        self.assertEqual([item.candidate_id for item in second_history], ["C0001/r0"])
        state = json.loads(
            (self.root / f"runs/searches/{result.search_id}/state.json").read_text()
        )
        self.assertEqual(len(state["curations"]), 2)
        self.assertEqual(state["active_memory"][0]["id"], "L0001")

    def test_provider_failure_retries_accumulated_evidence_and_does_not_block_search(self) -> None:
        worker = CapturingWorker(["runtime:0.8", "runtime:0.7"])
        curator = ScriptedCurator(
            [
                ProviderError("offline"),
                curator_response(["C0001/r0", "C0002/r0"]),
            ]
        )

        result = self.controller(worker, curator).run_search(
            self.baseline, self.baseline_evaluation
        )

        self.assertEqual(curator.evidence_sizes, [1, 2])
        self.assertEqual([step.status for step in result.curations], ["provider_failed", "completed"])
        self.assertEqual(result.best_candidate.revision_id, "C0002/r0")
        self.assertEqual(result.candidates_consumed, 2)


if __name__ == "__main__":
    unittest.main()

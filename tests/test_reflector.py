from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.llm import ProviderError
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.reflector import ReflectionError, Reflector, RevisionEvidence
from kernelbunny.worker import Worker


def worker_proposal(source: str, intent: str = "Change the test source.") -> str:
    return json.dumps({"worker_intent": intent, "cuda_source": source})


def reflection_response(confidence: str = "medium") -> str:
    return json.dumps(
        {
            "what_changed": "The candidate changed the kernel body.",
            "observed": "Compilation and held-out correctness passed.",
            "hypothesis": "The change may improve memory reuse.",
            "confidence": confidence,
        }
    )


class RecordingLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ScriptedWorker:
    def __init__(self, initial: str | Exception, repairs: list[str] | None = None) -> None:
        self.initial = initial
        self.repairs = list(repairs or [])

    def request_proposal(self, *args, **kwargs) -> str:
        if isinstance(self.initial, Exception):
            raise self.initial
        return self.initial

    def request_repair(self, *args, **kwargs) -> str:
        return self.repairs.pop(0)

    parse_proposal = staticmethod(Worker.parse_proposal)


class SourceEvaluator:
    def evaluate(self, candidate: Candidate) -> Evaluation:
        source = Path(candidate.source_path).read_text(encoding="utf-8")
        if source == "compile_bad":
            return Evaluation(
                candidate.revision_id,
                False,
                "candidate.cu:7: syntax error",
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
        return Evaluation(
            candidate.revision_id,
            True,
            "compiled",
            True,
            "Passed M=17, N=19, K=13, seed=37",
            0.5,
            "M=512, N=512, K=512; median=0.500000 ms",
        )


class ReflectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        parent_path = self.root / "baseline.cu"
        parent_path.write_text("parent source\n", encoding="utf-8")
        candidate_path = self.root / "candidate.cu"
        candidate_path.write_text("candidate source\n", encoding="utf-8")
        self.parent = Candidate("baseline", None, str(parent_path))
        self.parent_evaluation = Evaluation(
            "baseline/r0", True, "compiled", True, "hidden parent cases", 1.0,
            "M=512 benchmark",
        )
        self.candidate = Candidate(
            "C0001", "baseline/r0", str(candidate_path), "Improve memory reuse."
        )
        self.candidate_evaluation = Evaluation(
            "C0001/r0", True, "compiled", True,
            "Passed M=17, N=19, K=13, seed=37", 0.8,
            "M=512, N=512, K=512; median=0.8 ms",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_parser_assigns_trusted_candidate_id(self) -> None:
        reflection = Reflector.parse_reflection(
            reflection_response("HIGH"),
            "C0042/r2",
        )

        self.assertEqual(reflection.candidate_id, "C0042/r2")
        self.assertEqual(reflection.confidence, "high")

    def test_parser_rejects_extra_fields_and_invalid_confidence(self) -> None:
        payload = json.loads(reflection_response())
        payload["candidate_id"] = "model-controlled"
        with self.assertRaises(ReflectionError):
            Reflector.parse_reflection(json.dumps(payload), "C0001/r0")

        with self.assertRaises(ReflectionError):
            Reflector.parse_reflection(reflection_response("certain"), "C0001/r0")

    def test_prompt_contains_evidence_but_hides_trusted_inputs(self) -> None:
        llm = RecordingLLM(reflection_response())
        reflector = Reflector(llm)

        reflector.request_reflection(
            original_parent=self.parent,
            parent_evaluation=self.parent_evaluation,
            final_candidate=self.candidate,
            terminal_status="evaluated",
            history=(
                RevisionEvidence(self.candidate, self.candidate_evaluation, "initial"),
            ),
        )

        prompt = llm.prompts[0]
        self.assertIn("Improve memory reuse.", prompt)
        self.assertIn("20.00% faster than the parent", prompt)
        self.assertIn("passed held-out correctness", prompt)
        self.assertIn("candidate source", prompt)
        self.assertIn("Unified source diff", prompt)
        self.assertNotIn("M=17", prompt)
        self.assertNotIn("seed=37", prompt)
        self.assertNotIn("M=512", prompt)


class ControllerReflectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        parent_path = self.root / "baseline.cu"
        parent_path.write_text("good", encoding="utf-8")
        self.parent = Candidate("baseline", None, str(parent_path))
        self.parent_evaluation = Evaluation(
            "baseline/r0", True, "compiled", True, "correct", 1.0, "benchmarked"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_candidate(
        self,
        worker: ScriptedWorker,
        llm: RecordingLLM,
        config: ControllerConfig | None = None,
    ):
        controller = Controller(
            worker,
            SourceEvaluator(),
            self.root / "runs",
            config or ControllerConfig(
                max_compile_repairs=1,
                max_correctness_repairs=1,
            ),
            reflector=Reflector(llm),
        )
        return controller.run_candidate(self.parent, self.parent_evaluation)

    def test_repair_lifecycle_gets_one_terminal_reflection(self) -> None:
        llm = RecordingLLM(reflection_response())
        result = self.run_candidate(
            ScriptedWorker(
                worker_proposal("compile_bad"),
                [worker_proposal("good", "Repair the syntax error.")],
            ),
            llm,
        )

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.final_revision_id, "C0001/r1")
        self.assertEqual(result.reflection_status, "completed")
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("C0001/r0 (initial)", llm.prompts[0])
        self.assertIn("C0001/r1 (compilation_repair)", llm.prompts[0])
        self.assertFalse((self.root / "runs/C0001/r0/reflection.md").exists())
        self.assertTrue((self.root / "runs/C0001/r1/reflection.md").is_file())
        self.assertTrue((self.root / "runs/C0001/r1/reflection.json").is_file())
        metadata = json.loads(
            (self.root / "runs/C0001/metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["reflection_status"], "completed")

    def test_reflection_provider_failure_does_not_change_candidate_result(self) -> None:
        result = self.run_candidate(
            ScriptedWorker(worker_proposal("good")),
            RecordingLLM(ProviderError("reflector offline")),
        )

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.evaluation.runtime_ms, 0.5)
        self.assertEqual(result.reflection_status, "provider_failed")
        self.assertIn("reflector offline", result.reflection_error)
        self.assertTrue(
            (self.root / "runs/C0001/r0/reflection_error.txt").is_file()
        )

    def test_malformed_reflection_is_archived_but_nonfatal(self) -> None:
        result = self.run_candidate(
            ScriptedWorker(worker_proposal("good")),
            RecordingLLM("not json"),
        )

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.reflection_status, "parse_failed")
        self.assertTrue(
            (self.root / "runs/C0001/r0/reflection_raw_response.txt").is_file()
        )

    def test_no_source_means_reflection_is_not_applicable(self) -> None:
        llm = RecordingLLM(reflection_response())
        result = self.run_candidate(
            ScriptedWorker(ProviderError("worker offline")),
            llm,
        )

        self.assertEqual(result.status, "provider_failed")
        self.assertEqual(result.reflection_status, "not_applicable")
        self.assertEqual(llm.prompts, [])

    def test_terminal_compile_and_correctness_failures_are_reflected(self) -> None:
        for source, expected_status in (
            ("compile_bad", "compile_repair_exhausted"),
            ("wrong", "correctness_repair_exhausted"),
        ):
            with self.subTest(source=source):
                run_root = self.root / source
                run_root.mkdir()
                llm = RecordingLLM(reflection_response("low"))
                controller = Controller(
                    ScriptedWorker(worker_proposal(source)),
                    SourceEvaluator(),
                    run_root / "runs",
                    ControllerConfig(
                        max_compile_repairs=0,
                        max_correctness_repairs=0,
                    ),
                    reflector=Reflector(llm),
                )

                result = controller.run_candidate(
                    self.parent,
                    self.parent_evaluation,
                )

                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.reflection_status, "completed")
                self.assertEqual(len(llm.prompts), 1)
                self.assertIn(expected_status, llm.prompts[0])
                self.assertNotIn("M=17", llm.prompts[0])

    def test_reflection_failure_does_not_block_best_ever_or_selection(self) -> None:
        llm = RecordingLLM(ProviderError("reflector offline"))
        controller = Controller(
            ScriptedWorker(worker_proposal("good")),
            SourceEvaluator(),
            self.root / "search-runs",
            ControllerConfig(
                max_compile_repairs=0,
                max_correctness_repairs=0,
                population_size=2,
                children_per_parent=1,
                max_generations=1,
                max_candidates=1,
                max_no_improvement_generations=1,
            ),
            reflector=Reflector(llm),
        )

        result = controller.run_search(self.parent, self.parent_evaluation)
        attempt = result.steps[0].attempts[0].candidate_result

        self.assertEqual(attempt.reflection_status, "provider_failed")
        self.assertEqual(result.best_candidate.revision_id, "C0001/r0")
        self.assertEqual(result.best_evaluation.runtime_ms, 0.5)
        self.assertEqual(result.steps[0].selected_revision_ids[0], "C0001/r0")

    def test_malformed_initial_proposal_is_not_reflected(self) -> None:
        llm = RecordingLLM(reflection_response())
        result = self.run_candidate(ScriptedWorker("not json"), llm)

        self.assertEqual(result.status, "proposal_failed")
        self.assertEqual(result.reflection_status, "not_applicable")
        self.assertEqual(llm.prompts, [])


if __name__ == "__main__":
    unittest.main()

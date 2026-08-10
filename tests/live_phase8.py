"""Opt-in Phase 8 test using real DeepSeek calls and CUDA evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import re

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.curator import Curator
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import DeepSeekLLM, LLM, load_env_file
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.reflector import Reflector
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LIVE_ROOT = PROJECT_ROOT / "runs" / "phase8-live"
FIXTURE_LESSON = (
    "Preserve arbitrary-dimension boundary handling while optimizing memory access."
)


class RecordingLLM:
    def __init__(self, delegate: LLM) -> None:
        self.delegate = delegate
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.delegate.generate(prompt)


def _is_valid(evaluation: Evaluation | None) -> bool:
    return bool(
        evaluation
        and evaluation.compile_success
        and evaluation.correctness_success is True
        and evaluation.runtime_ms is not None
    )


def _reserve_session() -> Path:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in LIVE_ROOT.iterdir():
        match = re.fullmatch(r"T(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    session = LIVE_ROOT / f"T{max(numbers, default=0) + 1:04d}"
    session.mkdir(exist_ok=False)
    return session


def _seed_test_memory(memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=False)
    (memory_dir / "global_memory.json").write_text(
        json.dumps(
            {
                "lessons": [
                    {
                        "id": "L0001",
                        "text": FIXTURE_LESSON,
                        "evidence_candidate_ids": ["phase8-test-fixture/r0"],
                        "status": "active",
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    load_env_file(PROJECT_ROOT / ".env")
    session = _reserve_session()
    runs_dir = session / "candidates"
    memory_dir = session / "memory"
    _seed_test_memory(memory_dir)

    evaluator = Evaluator()
    baseline = Candidate(
        "baseline",
        None,
        str(BASELINE_PATH),
        "Trusted naive starting kernel.",
    )
    baseline_evaluation = evaluator.evaluate(baseline)
    if not _is_valid(baseline_evaluation):
        print("Live Phase 8 test requires a working CUDA baseline.")
        print(baseline_evaluation.compile_log)
        print(baseline_evaluation.correctness_log)
        return 1

    llm = RecordingLLM(DeepSeekLLM())
    controller = Controller(
        worker=Worker(llm),
        evaluator=evaluator,
        runs_dir=runs_dir,
        config=ControllerConfig(
            max_compile_repairs=1,
            max_correctness_repairs=1,
            population_size=2,
            children_per_parent=1,
            max_generations=1,
            max_candidates=1,
            max_no_improvement_generations=1,
            meaningful_improvement_ratio=0.02,
            curation_interval_generations=1,
            max_active_memory_lessons=10,
            max_curator_lessons_per_run=5,
            large_improvement_ratio=0.10,
            max_branch_reflections=3,
        ),
        reflector=Reflector(llm),
        curator=Curator(llm),
        memory_dir=memory_dir,
    )
    result = controller.run_search(baseline, baseline_evaluation)
    attempt = result.steps[0].attempts[0].candidate_result
    curation = result.curations[0]

    print(f"session={session}")
    print(f"baseline_runtime_ms={baseline_evaluation.runtime_ms}")
    print(
        f"candidate={attempt.final_revision_id} status={attempt.status} "
        f"runtime_ms={attempt.evaluation.runtime_ms if attempt.evaluation else None} "
        f"compile_repairs={attempt.compile_repairs_used} "
        f"correctness_repairs={attempt.correctness_repairs_used} "
        f"reflection_status={attempt.reflection_status}"
    )
    if attempt.final_candidate is not None:
        print(f"worker_intent={attempt.final_candidate.worker_intent}")
    if attempt.reflection is not None:
        print(f"reflection_observed={attempt.reflection.observed}")
        print(f"reflection_hypothesis={attempt.reflection.hypothesis}")
        print(f"reflection_confidence={attempt.reflection.confidence}")
    print(
        f"curation_status={curation.status} evidence={curation.evidence_count} "
        f"accepted={curation.accepted_lesson_ids}"
    )
    for lesson in result.active_memory:
        print(
            f"memory={lesson.id} status={lesson.status} "
            f"evidence={lesson.evidence_candidate_ids} text={lesson.text}"
        )

    worker_prompts = [
        prompt for prompt in llm.prompts if prompt.startswith("You are proposing")
    ]
    reflector_prompts = [
        prompt for prompt in llm.prompts if prompt.startswith("You are the reflector")
    ]
    curator_prompts = [
        prompt for prompt in llm.prompts if prompt.startswith("You curate compact")
    ]
    curation_dir = (
        runs_dir / "searches" / result.search_id / "curations" / "K0001"
    )
    memory_payload = json.loads(
        (memory_dir / "global_memory.json").read_text(encoding="utf-8")
    )
    checks = (
        _is_valid(attempt.evaluation),
        attempt.reflection_status == "completed",
        curation.status == "completed",
        len(worker_prompts) >= 1 and FIXTURE_LESSON in worker_prompts[0],
        len(reflector_prompts) == 1,
        len(curator_prompts) == 1,
        "candidate.cu" not in curator_prompts[0],
        "seed=" not in curator_prompts[0],
        (curation_dir / "raw_response.txt").is_file(),
        (curation_dir / "proposals.json").is_file(),
        (memory_dir / "global_memory.md").is_file(),
        (memory_dir / "curation_archive.jsonl").is_file(),
        len(memory_payload["lessons"]) >= 1,
        result.candidates_consumed == 1,
    )
    if not all(checks):
        print(f"checks={checks}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

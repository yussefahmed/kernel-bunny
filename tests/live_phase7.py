"""Opt-in Phase 7 lifecycle using real DeepSeek calls and CUDA evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import DeepSeekLLM, load_env_file
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.reflector import Reflector
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LIVE_RUNS = PROJECT_ROOT / "runs" / "phase7-live"


def _is_valid(evaluation: Evaluation | None) -> bool:
    return bool(
        evaluation
        and evaluation.compile_success
        and evaluation.correctness_success is True
        and evaluation.runtime_ms is not None
    )


def main() -> int:
    load_env_file(PROJECT_ROOT / ".env")
    evaluator = Evaluator()
    baseline = Candidate(
        "baseline",
        None,
        str(BASELINE_PATH),
        "Trusted naive starting kernel.",
    )
    baseline_evaluation = evaluator.evaluate(baseline)
    if not _is_valid(baseline_evaluation):
        print("Live Phase 7 test requires a working CUDA baseline.")
        print(baseline_evaluation.compile_log)
        print(baseline_evaluation.correctness_log)
        return 1

    llm = DeepSeekLLM()
    controller = Controller(
        worker=Worker(llm),
        evaluator=evaluator,
        runs_dir=LIVE_RUNS,
        config=ControllerConfig(
            max_compile_repairs=1,
            max_correctness_repairs=1,
            population_size=2,
            children_per_parent=1,
            max_generations=1,
            max_candidates=1,
            max_no_improvement_generations=1,
            meaningful_improvement_ratio=0.02,
        ),
        reflector=Reflector(llm),
    )
    result = controller.run_search(baseline, baseline_evaluation)
    attempt = result.steps[0].attempts[0].candidate_result

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
        print(f"what_changed={attempt.reflection.what_changed}")
        print(f"observed={attempt.reflection.observed}")
        print(f"hypothesis={attempt.reflection.hypothesis}")
        print(f"confidence={attempt.reflection.confidence}")
    elif attempt.reflection_error:
        print(f"reflection_error={attempt.reflection_error}")

    state_path = LIVE_RUNS / "searches" / result.search_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    final_candidate = attempt.final_candidate
    revision_dir = (
        Path(final_candidate.source_path).parent if final_candidate is not None else None
    )
    if revision_dir is not None:
        print(f"artifacts={revision_dir}")

    reflection_complete = (
        attempt.reflection_status == "completed"
        and attempt.reflection is not None
        and attempt.reflection.candidate_id == attempt.final_revision_id
    )
    artifacts_complete = bool(
        revision_dir
        and (revision_dir / "reflection_raw_response.txt").is_file()
        and (revision_dir / "reflection.json").is_file()
        and (revision_dir / "reflection.md").is_file()
    )
    state_matches = (
        state["status"] == "stopped"
        and state["stop_reason"] == "max_candidates"
        and state["generations"][0]["attempts"][0]["reflection_status"]
        == "completed"
    )
    trusted_result_survived = (
        attempt.status == "evaluated"
        and _is_valid(attempt.evaluation)
        and result.candidates_consumed == 1
    )
    checks = (
        reflection_complete,
        artifacts_complete,
        state_matches,
        trusted_result_survived,
    )
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

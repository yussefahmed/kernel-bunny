"""Opt-in Phase 5 search using the real DeepSeek API and CUDA evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import DeepSeekLLM, load_env_file
from kernelbunny.models import Candidate
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LIVE_RUNS = PROJECT_ROOT / "runs" / "phase5-live"


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
    if baseline_evaluation.runtime_ms is None:
        print("Live Phase 5 test requires a working CUDA baseline.")
        return 1

    controller = Controller(
        worker=Worker(DeepSeekLLM()),
        evaluator=evaluator,
        runs_dir=LIVE_RUNS,
        config=ControllerConfig(
            max_compile_repairs=1,
            max_correctness_repairs=1,
            max_generations=3,
            max_candidates=3,
            max_no_improvement_generations=3,
            meaningful_improvement_ratio=0.02,
        ),
    )
    result = controller.run_search(baseline, baseline_evaluation)

    print(f"baseline runtime_ms={baseline_evaluation.runtime_ms}")
    for step in result.steps:
        for attempt in step.attempts:
            candidate_result = attempt.candidate_result
            evaluation = candidate_result.evaluation
            runtime = evaluation.runtime_ms if evaluation else None
            print(
                f"generation={step.generation} parent={attempt.parent_revision_id} "
                f"candidate={candidate_result.final_revision_id} "
                f"status={candidate_result.status} runtime_ms={runtime} "
                f"compile_repairs={candidate_result.compile_repairs_used} "
                f"correctness_repairs={candidate_result.correctness_repairs_used} "
                f"meaningful={step.meaningful_improvement} "
                f"active={step.selected_revision_ids} best={step.best_revision_id}"
            )
    print(
        f"search={result.search_id} stop_reason={result.stop_reason} "
        f"generations={result.generations_attempted} "
        f"candidates={result.candidates_consumed} "
        f"no_improvement={result.no_improvement_generations} "
        f"best={result.best_candidate.revision_id} "
        f"best_runtime_ms={result.best_evaluation.runtime_ms}"
    )

    state_path = LIVE_RUNS / "searches" / result.search_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    valid_stop_reasons = {
        "max_candidates",
        "no_meaningful_improvement",
        "max_generations",
    }
    state_matches = (
        state["status"] == "stopped"
        and state["stop_reason"] == result.stop_reason
        and [entry["revision_id"] for entry in state["active_population"]]
        == [member.candidate.revision_id for member in result.active_population]
        and state["best_ever"]["revision_id"] == result.best_candidate.revision_id
    )
    bounds_hold = (
        result.generations_attempted <= 3
        and result.candidates_consumed <= 3
        and result.no_improvement_generations <= 3
        and len(result.steps) == result.generations_attempted
    )
    api_produced_candidate = result.candidates_consumed >= 1
    return 0 if (
        result.stop_reason in valid_stop_reasons
        and state_matches
        and bounds_hold
        and api_produced_candidate
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

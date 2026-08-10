"""Opt-in Phase 6 population search using the real DeepSeek API and CUDA."""

from __future__ import annotations

import json
from pathlib import Path

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import DeepSeekLLM, load_env_file
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LIVE_RUNS = PROJECT_ROOT / "runs" / "phase6-live"


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
        print("Live Phase 6 test requires a working CUDA baseline.")
        return 1

    controller = Controller(
        worker=Worker(DeepSeekLLM()),
        evaluator=evaluator,
        runs_dir=LIVE_RUNS,
        config=ControllerConfig(
            max_compile_repairs=1,
            max_correctness_repairs=1,
            population_size=3,
            children_per_parent=2,
            max_generations=2,
            max_candidates=8,
            max_no_improvement_generations=3,
            meaningful_improvement_ratio=0.02,
        ),
    )
    result = controller.run_search(baseline, baseline_evaluation)

    print(f"baseline runtime_ms={baseline_evaluation.runtime_ms}")
    valid_runtimes = [baseline_evaluation.runtime_ms]
    for step in result.steps:
        print(f"generation={step.generation} parents={step.parent_revision_ids}")
        for attempt in step.attempts:
            candidate_result = attempt.candidate_result
            evaluation = candidate_result.evaluation
            runtime = evaluation.runtime_ms if evaluation else None
            intent = (
                candidate_result.final_candidate.worker_intent
                if candidate_result.final_candidate
                else ""
            )
            print(
                f"  parent={attempt.parent_revision_id} child={attempt.child_number} "
                f"candidate={candidate_result.final_revision_id} "
                f"status={candidate_result.status} runtime_ms={runtime} "
                f"compile_repairs={candidate_result.compile_repairs_used} "
                f"correctness_repairs={candidate_result.correctness_repairs_used} "
                f"intent={intent}"
            )
            if _is_valid(evaluation):
                valid_runtimes.append(evaluation.runtime_ms)
        print(
            f"  selected={step.selected_revision_ids} "
            f"meaningful_improvement={step.meaningful_improvement} "
            f"best={step.best_revision_id}"
        )

    active_ids = [
        member.candidate.revision_id for member in result.active_population
    ]
    print(
        f"search={result.search_id} stop_reason={result.stop_reason} "
        f"generations={result.generations_attempted} "
        f"candidates={result.candidates_consumed} "
        f"no_improvement={result.no_improvement_generations} "
        f"active={active_ids} best={result.best_candidate.revision_id} "
        f"best_runtime_ms={result.best_evaluation.runtime_ms}"
    )

    state_path = LIVE_RUNS / "searches" / result.search_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    persisted_active_ids = [
        entry["revision_id"] for entry in state["active_population"]
    ]
    valid_population = (
        1 <= len(result.active_population) <= 3
        and len(active_ids) == len(set(active_ids))
        and all(_is_valid(member.evaluation) for member in result.active_population)
    )
    state_matches = (
        state["status"] == "stopped"
        and state["stop_reason"] == result.stop_reason
        and persisted_active_ids == active_ids
        and state["best_ever"]["revision_id"]
        == result.best_candidate.revision_id
        and len(state["generations"]) == result.generations_attempted
    )
    bounds_hold = (
        result.generations_attempted == 2
        and result.candidates_consumed <= 8
        and len(result.steps) == 2
    )
    best_is_correct = result.best_evaluation.runtime_ms == min(valid_runtimes)
    api_produced_candidate = result.candidates_consumed >= 1
    return 0 if all(
        (
            valid_population,
            state_matches,
            bounds_hold,
            best_is_correct,
            api_produced_candidate,
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

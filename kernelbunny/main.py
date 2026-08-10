"""Command-line entry point for the KernelBunny MVP."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from kernelbunny.controller import Controller, ControllerConfig, SearchResult
from kernelbunny.curator import Curator
from kernelbunny.evaluator import Evaluator, EvaluatorConfig
from kernelbunny.llm import DeepSeekConfig, DeepSeekLLM, load_env_file
from kernelbunny.models import Candidate, Evaluation
from kernelbunny.reflector import Reflector
from kernelbunny.trusted.benchmark import BenchmarkConfig
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
MEMORY_DIR = PROJECT_ROOT / "memory"
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LOGGER = logging.getLogger("kernelbunny")


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free CLI without starting CUDA or an LLM client."""

    parser = argparse.ArgumentParser(
        prog="kernelbunny",
        description="Propose, independently validate, and evolve CUDA SGEMM kernels.",
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--baseline-path", type=Path, default=BASELINE_PATH)
    common.add_argument("--nvcc", default="nvcc")
    common.add_argument("--gpu-arch", default="sm_86")
    common.add_argument(
        "--compile-timeout-seconds", type=_positive_float, default=60.0
    )
    common.add_argument(
        "--execution-timeout-seconds", type=_positive_float, default=30.0
    )
    common.add_argument("--warmup-runs", type=_positive_int, default=10)
    common.add_argument("--measured-runs", type=_positive_int, default=50)
    common.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    subparsers.add_parser(
        "baseline",
        parents=[common],
        help="Compile, validate, and benchmark the trusted baseline without an LLM.",
    )

    search = subparsers.add_parser(
        "search",
        parents=[common],
        help="Run the bounded LLM-guided evolutionary search.",
    )
    search.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    search.add_argument("--memory-dir", type=Path, default=MEMORY_DIR)
    search.add_argument("--population-size", type=_positive_int, default=3)
    search.add_argument("--children-per-parent", type=_positive_int, default=2)
    search.add_argument("--max-generations", type=_positive_int, default=6)
    search.add_argument("--max-candidates", type=_positive_int, default=12)
    search.add_argument(
        "--max-no-improvement-generations", type=_positive_int, default=3
    )
    search.add_argument("--max-compile-repairs", type=_nonnegative_int, default=1)
    search.add_argument("--max-correctness-repairs", type=_nonnegative_int, default=1)
    search.add_argument(
        "--meaningful-improvement-percent", type=_percentage, default=2.0
    )
    search.add_argument(
        "--curation-interval-generations", type=_positive_int, default=3
    )
    search.add_argument("--model", default="deepseek-v4-flash")
    search.add_argument(
        "--llm-timeout-seconds", type=_positive_float, default=120.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    _configure_logging(args.log_level)
    load_env_file(PROJECT_ROOT / ".env")

    evaluator = Evaluator(_evaluator_config_from_args(args))
    baseline = Candidate(
        id="baseline",
        parent_id=None,
        source_path=str(args.baseline_path.resolve()),
        worker_intent="Trusted naive starting kernel.",
    )
    baseline_evaluation = evaluator.evaluate(baseline)
    if not _is_valid(baseline_evaluation):
        _log_baseline_failure(baseline_evaluation)
        return 1

    LOGGER.info("Baseline validated: runtime_ms=%.6f", baseline_evaluation.runtime_ms)
    if args.command == "baseline":
        return 0

    llm = DeepSeekLLM(_deepseek_config_from_args(args))
    controller = Controller(
        worker=Worker(llm),
        evaluator=evaluator,
        runs_dir=args.runs_dir.resolve(),
        config=_controller_config_from_args(args),
        reflector=Reflector(llm),
        curator=Curator(llm),
        memory_dir=args.memory_dir.resolve(),
    )
    result = controller.run_search(baseline, baseline_evaluation)
    _log_search_result(result)
    return 0


def _evaluator_config_from_args(args: argparse.Namespace) -> EvaluatorConfig:
    return EvaluatorConfig(
        nvcc=args.nvcc,
        gpu_arch=args.gpu_arch,
        compile_timeout_seconds=args.compile_timeout_seconds,
        execution_timeout_seconds=args.execution_timeout_seconds,
        benchmark=BenchmarkConfig(
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        ),
    )


def _deepseek_config_from_args(args: argparse.Namespace) -> DeepSeekConfig:
    return DeepSeekConfig(model=args.model, timeout_seconds=args.llm_timeout_seconds)


def _controller_config_from_args(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        max_compile_repairs=args.max_compile_repairs,
        max_correctness_repairs=args.max_correctness_repairs,
        population_size=args.population_size,
        children_per_parent=args.children_per_parent,
        max_generations=args.max_generations,
        max_candidates=args.max_candidates,
        max_no_improvement_generations=args.max_no_improvement_generations,
        meaningful_improvement_ratio=args.meaningful_improvement_percent / 100.0,
        curation_interval_generations=args.curation_interval_generations,
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s %(message)s",
        force=True,
    )


def _log_baseline_failure(evaluation: Evaluation) -> None:
    LOGGER.error("Baseline evaluation failed; no worker was called.")
    if not evaluation.compile_success:
        LOGGER.error("Compilation: %s", evaluation.compile_log)
    elif evaluation.correctness_success is not True:
        LOGGER.error("Correctness failed in the trusted evaluator.")
    else:
        LOGGER.error("Benchmark failed in the trusted evaluator.")


def _log_search_result(result: SearchResult) -> None:
    for step in result.steps:
        LOGGER.info(
            "Generation %d: parents=%s",
            step.generation,
            ",".join(step.parent_revision_ids),
        )
        for attempt in step.attempts:
            candidate = attempt.candidate_result
            runtime = candidate.evaluation.runtime_ms if candidate.evaluation else None
            LOGGER.info(
                "Candidate %s parent=%s status=%s runtime_ms=%s "
                "compile_repairs=%d correctness_repairs=%d reflection=%s",
                candidate.final_revision_id or candidate.candidate_id,
                attempt.parent_revision_id,
                candidate.status,
                f"{runtime:.6f}" if runtime is not None else "n/a",
                candidate.compile_repairs_used,
                candidate.correctness_repairs_used,
                candidate.reflection_status,
            )
        LOGGER.info(
            "Generation %d selected=%s meaningful_improvement=%s best=%s",
            step.generation,
            ",".join(step.selected_revision_ids),
            step.meaningful_improvement,
            step.best_revision_id,
        )

    for curation in result.curations:
        LOGGER.info(
            "Curation generation=%d status=%s evidence=%d accepted=%s",
            curation.generation,
            curation.status,
            curation.evidence_count,
            ",".join(curation.accepted_lesson_ids) or "none",
        )

    LOGGER.info(
        "Search %s stopped: reason=%s generations=%d candidates=%d "
        "best=%s best_runtime_ms=%.6f active=%s",
        result.search_id,
        result.stop_reason,
        result.generations_attempted,
        result.candidates_consumed,
        result.best_candidate.revision_id,
        result.best_evaluation.runtime_ms,
        ",".join(member.candidate.revision_id for member in result.active_population),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 100.0:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def _is_valid(evaluation: Evaluation) -> bool:
    return (
        evaluation.compile_success
        and evaluation.correctness_success is True
        and evaluation.runtime_ms is not None
    )


if __name__ == "__main__":
    raise SystemExit(main())

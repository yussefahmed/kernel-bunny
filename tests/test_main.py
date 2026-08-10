"""Unit tests for the safe Phase 9 command-line entry point."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from kernelbunny.controller import PopulationMember, SearchResult
from kernelbunny.main import (
    _controller_config_from_args,
    _evaluator_config_from_args,
    build_parser,
    main,
)
from kernelbunny.models import Candidate, Evaluation


def valid_baseline() -> tuple[Candidate, Evaluation]:
    candidate = Candidate("baseline", None, "/tmp/baseline.cu")
    evaluation = Evaluation(
        "baseline/r0",
        True,
        "compiled",
        True,
        "correct",
        1.25,
        "benchmarked",
    )
    return candidate, evaluation


class CliTests(unittest.TestCase):
    def test_no_command_prints_help_without_constructing_services(self) -> None:
        output = io.StringIO()
        with (
            patch("kernelbunny.main.Evaluator") as evaluator,
            patch("kernelbunny.main.DeepSeekLLM") as llm,
            redirect_stdout(output),
        ):
            result = main([])

        self.assertEqual(result, 0)
        self.assertIn("{baseline,search}", output.getvalue())
        evaluator.assert_not_called()
        llm.assert_not_called()

    def test_baseline_command_never_constructs_an_llm(self) -> None:
        _, evaluation = valid_baseline()
        with (
            patch("kernelbunny.main.Evaluator") as evaluator_type,
            patch("kernelbunny.main.DeepSeekLLM") as llm,
            redirect_stderr(io.StringIO()),
        ):
            evaluator_type.return_value.evaluate.return_value = evaluation
            result = main(["baseline"])

        self.assertEqual(result, 0)
        evaluator_type.return_value.evaluate.assert_called_once()
        llm.assert_not_called()

    def test_cli_values_map_to_trusted_configs(self) -> None:
        args = build_parser().parse_args(
            [
                "search",
                "--gpu-arch",
                "sm_90",
                "--warmup-runs",
                "4",
                "--measured-runs",
                "12",
                "--population-size",
                "2",
                "--children-per-parent",
                "1",
                "--max-generations",
                "2",
                "--max-candidates",
                "3",
                "--max-compile-repairs",
                "0",
                "--meaningful-improvement-percent",
                "3.5",
            ]
        )

        evaluator = _evaluator_config_from_args(args)
        controller = _controller_config_from_args(args)
        self.assertEqual(evaluator.gpu_arch, "sm_90")
        self.assertEqual(evaluator.benchmark.warmup_runs, 4)
        self.assertEqual(evaluator.benchmark.measured_runs, 12)
        self.assertEqual(controller.population_size, 2)
        self.assertEqual(controller.children_per_parent, 1)
        self.assertEqual(controller.max_generations, 2)
        self.assertEqual(controller.max_candidates, 3)
        self.assertEqual(controller.max_compile_repairs, 0)
        self.assertAlmostEqual(controller.meaningful_improvement_ratio, 0.035)

    def test_invalid_budget_is_rejected_by_argparse(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(["search", "--max-candidates", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_failed_baseline_stops_before_llm_or_controller(self) -> None:
        failure = Evaluation(
            "baseline/r0",
            True,
            "compiled",
            False,
            "failed",
            None,
            "not run",
        )
        with (
            patch("kernelbunny.main.Evaluator") as evaluator_type,
            patch("kernelbunny.main.DeepSeekLLM") as llm,
            patch("kernelbunny.main.Controller") as controller,
            redirect_stderr(io.StringIO()),
        ):
            evaluator_type.return_value.evaluate.return_value = failure
            result = main(["search"])

        self.assertEqual(result, 1)
        llm.assert_not_called()
        controller.assert_not_called()

    def test_successful_search_logs_trusted_final_summary(self) -> None:
        _, evaluation = valid_baseline()
        actual_baseline = Candidate(
            "baseline",
            None,
            str(Path("kernels/baseline.cu").resolve()),
            "Trusted naive starting kernel.",
        )
        result = SearchResult(
            search_id="S0001",
            stop_reason="max_candidates",
            generations_attempted=1,
            candidates_consumed=1,
            no_improvement_generations=0,
            active_population=(PopulationMember(actual_baseline, evaluation),),
            best_candidate=actual_baseline,
            best_evaluation=evaluation,
            steps=(),
            curations=(),
            active_memory=(),
        )
        logs = io.StringIO()
        with (
            patch("kernelbunny.main.Evaluator") as evaluator_type,
            patch("kernelbunny.main.DeepSeekLLM"),
            patch("kernelbunny.main.Controller") as controller_type,
            redirect_stderr(logs),
        ):
            evaluator_type.return_value.evaluate.return_value = evaluation
            controller_type.return_value.run_search.return_value = result
            exit_code = main(["search", "--max-candidates", "1"])

        self.assertEqual(exit_code, 0)
        self.assertIn("reason=max_candidates", logs.getvalue())
        self.assertIn("best=baseline/r0", logs.getvalue())


if __name__ == "__main__":
    unittest.main()

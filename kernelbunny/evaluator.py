"""Fail-fast orchestration for trusted CUDA candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile

from kernelbunny.models import Candidate, Evaluation
from kernelbunny.trusted.benchmark import BenchmarkConfig, run_benchmark
from kernelbunny.trusted.correctness import run_correctness


@dataclass(frozen=True)
class EvaluatorConfig:
    nvcc: str = "nvcc"
    gpu_arch: str = "sm_86"
    compile_timeout_seconds: float = 60.0
    execution_timeout_seconds: float = 30.0
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


class Evaluator:
    """Compile, validate, then benchmark exactly one candidate source."""

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self.config = config or EvaluatorConfig()
        self._runner_path = Path(__file__).parent / "trusted" / "runner.cu"

    def evaluate(self, candidate: Candidate) -> Evaluation:
        source_path = Path(candidate.source_path).resolve()
        if not source_path.is_file():
            return Evaluation(
                candidate_id=candidate.revision_id,
                compile_success=False,
                compile_log=f"Candidate source does not exist: {source_path}",
                correctness_success=None,
                correctness_log="Not run because compilation did not succeed.",
                runtime_ms=None,
                benchmark_log="Not run because correctness was not established.",
            )

        with tempfile.TemporaryDirectory(
            prefix=f"kernelbunny-{candidate.id}-r{candidate.revision}-"
        ) as build_dir:
            executable = Path(build_dir) / "candidate_runner"
            compile_success, compile_log = self._compile(source_path, executable)
            if not compile_success:
                return Evaluation(
                    candidate_id=candidate.revision_id,
                    compile_success=False,
                    compile_log=compile_log,
                    correctness_success=None,
                    correctness_log="Not run because compilation did not succeed.",
                    runtime_ms=None,
                    benchmark_log="Not run because correctness was not established.",
                )

            correctness = run_correctness(
                executable,
                timeout_seconds=self.config.execution_timeout_seconds,
            )
            if not correctness.success:
                return Evaluation(
                    candidate_id=candidate.revision_id,
                    compile_success=True,
                    compile_log=compile_log,
                    correctness_success=False,
                    correctness_log=correctness.log,
                    runtime_ms=None,
                    benchmark_log="Not run because correctness failed.",
                )

            benchmark = run_benchmark(
                executable,
                config=self.config.benchmark,
                timeout_seconds=self.config.execution_timeout_seconds,
            )
            return Evaluation(
                candidate_id=candidate.revision_id,
                compile_success=True,
                compile_log=compile_log,
                correctness_success=True,
                correctness_log=correctness.log,
                runtime_ms=benchmark.runtime_ms,
                benchmark_log=benchmark.log,
            )

    def _compile(self, source_path: Path, executable: Path) -> tuple[bool, str]:
        command = [
            self.config.nvcc,
            "-std=c++17",
            "-O3",
            f"-arch={self.config.gpu_arch}",
            str(self._runner_path),
            str(source_path),
            "-o",
            str(executable),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.compile_timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return False, f"CUDA compiler not found: {self.config.nvcc}"
        except subprocess.TimeoutExpired:
            return False, f"Compilation timed out after {self.config.compile_timeout_seconds:g}s."

        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        if completed.returncode != 0:
            return False, output[-16_000:] or f"nvcc exited with code {completed.returncode}."
        return True, output[-16_000:] or "Compilation succeeded."

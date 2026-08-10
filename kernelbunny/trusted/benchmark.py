"""Trusted SGEMM benchmark invocation and robust runtime aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import subprocess


@dataclass(frozen=True)
class BenchmarkConfig:
    m: int = 512
    n: int = 512
    k: int = 512
    seed: int = 67
    warmup_runs: int = 10
    measured_runs: int = 50


@dataclass(frozen=True)
class BenchmarkResult:
    runtime_ms: float | None
    log: str


def run_benchmark(
    executable: Path,
    *,
    config: BenchmarkConfig,
    timeout_seconds: float,
) -> BenchmarkResult:
    """Benchmark a validated candidate and return the median sample."""

    command = [
        str(executable),
        "benchmark",
        str(config.m),
        str(config.n),
        str(config.k),
        str(config.seed),
        str(config.warmup_runs),
        str(config.measured_runs),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            runtime_ms=None,
            log=f"Timed out after {timeout_seconds:g}s.",
        )

    protocol = _parse_protocol(completed.stdout)
    if completed.returncode != 0 or protocol.get("KB_STATUS") != "PASS":
        return BenchmarkResult(None, _process_details(completed))

    raw_times = protocol.get("KB_TIMES_MS", "")
    try:
        samples = [float(value) for value in raw_times.split(",") if value]
    except ValueError:
        return BenchmarkResult(None, "Runner returned malformed timing samples.")

    if len(samples) != config.measured_runs:
        return BenchmarkResult(
            None,
            f"Expected {config.measured_runs} timing samples, received {len(samples)}.",
        )
    if any(not math.isfinite(sample) or sample <= 0.0 for sample in samples):
        return BenchmarkResult(None, "Runner returned a non-positive or non-finite timing.")

    median_ms = statistics.median(samples)
    return BenchmarkResult(
        runtime_ms=median_ms,
        log=(
            f"M={config.m}, N={config.n}, K={config.k}; "
            f"warmups={config.warmup_runs}; samples={config.measured_runs}; "
            f"median={median_ms:.6f} ms; min={min(samples):.6f} ms; "
            f"max={max(samples):.6f} ms"
        ),
    )


def _parse_protocol(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith("KB_"):
            values[key] = value.strip()
    return values


def _process_details(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return output[-8_000:] if output else f"Runner exited with code {completed.returncode}."

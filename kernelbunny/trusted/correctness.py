"""Invoke the trusted runner against held-out SGEMM cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable

@dataclass(frozen=True)
class CorrectnessResult:
    success: bool
    log: str


@dataclass(frozen=True)
class GemmCase:
    m: int
    n: int
    k: int
    seed: int

    @property
    def label(self) -> str:
        return f"M={self.m}, N={self.n}, K={self.k}, seed={self.seed}"


# These deliberately include tiny and non-tile-aligned dimensions. Workers will
# receive the supported input contract, not this exact held-out case list.
CORRECTNESS_CASES = (
    GemmCase(m=1, n=1, k=1, seed=11),
    GemmCase(m=3, n=5, k=7, seed=23),
    GemmCase(m=17, n=19, k=13, seed=37),
    GemmCase(m=31, n=37, k=29, seed=41),
    GemmCase(m=64, n=48, k=80, seed=53),
)


def run_correctness(
    executable: Path,
    *,
    timeout_seconds: float,
    cases: Iterable[GemmCase] = CORRECTNESS_CASES,
) -> CorrectnessResult:
    """Run each trusted case, stopping at the first failure."""

    passed: list[str] = []
    for case in cases:
        command = [
            str(executable),
            "correctness",
            str(case.m),
            str(case.n),
            str(case.k),
            str(case.seed),
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
            return CorrectnessResult(
                success=False,
                log=f"Timed out after {timeout_seconds:g}s for {case.label}.",
            )

        protocol = _parse_protocol(completed.stdout)
        if completed.returncode != 0 or protocol.get("KB_STATUS") != "PASS":
            details = _process_details(completed)
            return CorrectnessResult(
                success=False,
                log=f"Failed {case.label}.\n{details}",
            )

        passed.append(case.label)

    return CorrectnessResult(
        success=True,
        log="Passed trusted correctness cases:\n" + "\n".join(passed),
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

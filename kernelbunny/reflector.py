"""Separate LLM role for cautious post-hoc experiment interpretation."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
from pathlib import Path

from kernelbunny.llm import LLM
from kernelbunny.models import Candidate, Evaluation, Reflection


class ReflectionError(ValueError):
    """The model responded, but did not satisfy the reflection contract."""


@dataclass(frozen=True)
class RevisionEvidence:
    """One trusted evaluation in the logical candidate's repair history."""

    candidate: Candidate
    evaluation: Evaluation
    attempt_kind: str

    def __post_init__(self) -> None:
        if self.evaluation.candidate_id != self.candidate.revision_id:
            raise ValueError("Revision evidence evaluation does not match its candidate.")


class Reflector:
    def __init__(self, llm: LLM, prompt_path: Path | None = None) -> None:
        self._llm = llm
        self._prompt_path = (
            prompt_path or Path(__file__).parent / "prompts" / "reflector.md"
        )

    def request_reflection(
        self,
        *,
        original_parent: Candidate,
        parent_evaluation: Evaluation,
        final_candidate: Candidate,
        terminal_status: str,
        history: tuple[RevisionEvidence, ...],
    ) -> str:
        """Request raw text while exposing only sanitized trusted observations."""

        if parent_evaluation.candidate_id != original_parent.revision_id:
            raise ValueError("Parent evaluation does not match the original parent.")
        if not history or history[-1].candidate != final_candidate:
            raise ValueError("Reflection history must end at the final candidate.")

        parent_source = Path(original_parent.source_path).read_text(encoding="utf-8")
        candidate_source = Path(final_candidate.source_path).read_text(encoding="utf-8")
        source_diff = "".join(
            difflib.unified_diff(
                parent_source.splitlines(keepends=True),
                candidate_source.splitlines(keepends=True),
                fromfile=original_parent.revision_id,
                tofile=final_candidate.revision_id,
            )
        )
        if not source_diff:
            source_diff = "No textual source change."

        prompt = self._prompt_path.read_text(encoding="utf-8")
        prompt += (
            f"\n\nOriginal parent: {original_parent.revision_id}"
            f"\nFinal candidate: {final_candidate.revision_id}"
            f"\nTerminal lifecycle status: {terminal_status}"
            f"\nWorker intent: {final_candidate.worker_intent or 'Not supplied.'}"
            f"\nTrusted parent runtime: {_runtime_text(parent_evaluation.runtime_ms)}"
            f"\nTrusted final runtime: {_runtime_text(history[-1].evaluation.runtime_ms)}"
            "\nTrusted runtime change: "
            f"{_runtime_delta_text(parent_evaluation, history[-1].evaluation)}"
            "\nAvailable compiler/GPU metrics: none collected."
            "\n\nSanitized revision history:\n"
            + "\n".join(_evidence_line(item) for item in history)
            + "\n\nOriginal parent CUDA source:\n```cuda\n"
            + parent_source
            + "\n```\n\nFinal candidate CUDA source:\n```cuda\n"
            + candidate_source
            + "\n```\n\nUnified source diff:\n```diff\n"
            + source_diff
            + "\n```\n"
        )
        return self._llm.generate(prompt)

    @staticmethod
    def parse_reflection(raw_response: str, candidate_id: str) -> Reflection:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ReflectionError(
                f"Reflector response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ReflectionError("Reflector response must be a JSON object.")
        expected_fields = {"what_changed", "observed", "hypothesis", "confidence"}
        if set(payload) != expected_fields:
            raise ReflectionError(
                "Reflector response must contain exactly what_changed, observed, "
                "hypothesis, and confidence."
            )

        values = {name: payload[name] for name in expected_fields}
        if any(
            not isinstance(value, str) or not value.strip()
            for value in values.values()
        ):
            raise ReflectionError("Every reflection field must be a non-empty string.")
        confidence = values["confidence"].strip().lower()
        if confidence not in {"low", "medium", "high"}:
            raise ReflectionError("confidence must be low, medium, or high.")
        return Reflection(
            candidate_id=candidate_id,
            what_changed=values["what_changed"].strip(),
            observed=values["observed"].strip(),
            hypothesis=values["hypothesis"].strip(),
            confidence=confidence,
        )


def render_reflection(reflection: Reflection, worker_intent: str) -> str:
    return (
        f"# Reflection: {reflection.candidate_id}\n\n"
        f"## Worker intent\n\n{worker_intent or 'Not supplied.'}\n\n"
        f"## What changed\n\n{reflection.what_changed}\n\n"
        f"## Observed\n\n{reflection.observed}\n\n"
        f"## Hypothesis\n\n{reflection.hypothesis}\n\n"
        f"## Confidence\n\n{reflection.confidence.capitalize()}\n"
    )


def _evidence_line(item: RevisionEvidence) -> str:
    evaluation = item.evaluation
    if evaluation.correctness_success is True:
        correctness = "passed held-out correctness"
    elif evaluation.correctness_success is False:
        correctness = "failed a held-out correctness case"
    else:
        correctness = "correctness not run"
    compiler_diagnostic = ""
    if not evaluation.compile_success:
        compiler_diagnostic = f"; compiler diagnostic={evaluation.compile_log[-4_000:]}"
    return (
        f"- {item.candidate.revision_id} ({item.attempt_kind}): "
        f"compile={'passed' if evaluation.compile_success else 'failed'}; "
        f"{correctness}; runtime={_runtime_text(evaluation.runtime_ms)}"
        f"{compiler_diagnostic}"
    )


def _runtime_text(runtime_ms: float | None) -> str:
    return "not available" if runtime_ms is None else f"{runtime_ms:.6f} ms"


def _runtime_delta_text(parent: Evaluation, candidate: Evaluation) -> str:
    if parent.runtime_ms is None or candidate.runtime_ms is None:
        return "not available"
    percent = (candidate.runtime_ms - parent.runtime_ms) / parent.runtime_ms * 100.0
    if percent == 0.0:
        return "unchanged from the parent"
    direction = "slower" if percent > 0 else "faster"
    return f"{abs(percent):.2f}% {direction} than the parent"

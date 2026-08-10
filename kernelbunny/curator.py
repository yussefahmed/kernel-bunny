"""Cross-experiment LLM role that proposes advisory global lessons."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from kernelbunny.llm import LLM
from kernelbunny.models import MemoryLesson, Reflection


class CuratorError(ValueError):
    """The model responded, but did not satisfy the curator contract."""


@dataclass(frozen=True)
class CurationEvidence:
    candidate_id: str
    parent_id: str
    terminal_status: str
    compile_success: bool
    correctness_success: bool | None
    runtime_ms: float | None
    parent_runtime_ms: float
    reflection: Reflection

    def __post_init__(self) -> None:
        if self.reflection.candidate_id != self.candidate_id:
            raise ValueError("Curation evidence does not match its reflection.")
        if self.parent_runtime_ms <= 0.0:
            raise ValueError("Curation evidence requires a positive parent runtime.")

    @property
    def improvement_ratio(self) -> float | None:
        if (
            not self.compile_success
            or self.correctness_success is not True
            or self.runtime_ms is None
        ):
            return None
        return (self.parent_runtime_ms - self.runtime_ms) / self.parent_runtime_ms


@dataclass(frozen=True)
class CuratorProposal:
    existing_lesson_id: str | None
    lesson: str
    evidence_candidate_ids: tuple[str, ...]


class Curator:
    def __init__(self, llm: LLM, prompt_path: Path | None = None) -> None:
        self._llm = llm
        self._prompt_path = (
            prompt_path or Path(__file__).parent / "prompts" / "curator.md"
        )

    def request_curation(
        self,
        evidence: tuple[CurationEvidence, ...],
        existing_lessons: tuple[MemoryLesson, ...],
        max_lessons: int = 5,
    ) -> str:
        if not evidence:
            raise ValueError("Curator requires at least one reflection.")
        if max_lessons <= 0:
            raise ValueError("max_lessons must be positive.")

        prompt = self._prompt_path.read_text(encoding="utf-8")
        prompt += f"\n\nReturn at most {max_lessons} lessons."
        prompt += "\n\nExisting non-archived lessons:\n"
        if existing_lessons:
            prompt += "\n".join(
                f"- {lesson.id} [{lesson.status}]: {lesson.text} "
                f"(evidence: {', '.join(lesson.evidence_candidate_ids)})"
                for lesson in existing_lessons
            )
        else:
            prompt += "- None."

        prompt += "\n\nNew experiment evidence:\n"
        prompt += "\n".join(_evidence_text(item) for item in evidence)
        return self._llm.generate(prompt)

    @staticmethod
    def parse_proposals(
        raw_response: str,
        *,
        allowed_evidence_ids: set[str],
        allowed_existing_lesson_ids: set[str],
        max_lessons: int,
    ) -> tuple[CuratorProposal, ...]:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise CuratorError(f"Curator response is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict) or set(payload) != {"lessons"}:
            raise CuratorError("Curator response must contain exactly lessons.")
        lessons = payload["lessons"]
        if not isinstance(lessons, list):
            raise CuratorError("lessons must be a list.")
        if len(lessons) > max_lessons:
            raise CuratorError(f"Curator returned more than {max_lessons} lessons.")

        parsed: list[CuratorProposal] = []
        for item in lessons:
            expected = {"existing_lesson_id", "lesson", "evidence_candidate_ids"}
            if not isinstance(item, dict) or set(item) != expected:
                raise CuratorError(
                    "Each lesson must contain exactly existing_lesson_id, lesson, "
                    "and evidence_candidate_ids."
                )
            existing_id = item["existing_lesson_id"]
            if existing_id is not None and (
                not isinstance(existing_id, str)
                or existing_id not in allowed_existing_lesson_ids
            ):
                raise CuratorError("Curator referenced an unknown existing lesson.")
            text = item["lesson"]
            if not isinstance(text, str) or not text.strip():
                raise CuratorError("Curator lesson text must be non-empty.")
            evidence_ids = item["evidence_candidate_ids"]
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or any(not isinstance(value, str) for value in evidence_ids)
                or len(set(evidence_ids)) != len(evidence_ids)
            ):
                raise CuratorError("Each lesson requires unique evidence candidate ids.")
            if not set(evidence_ids).issubset(allowed_evidence_ids):
                raise CuratorError("Curator referenced evidence outside its input window.")
            parsed.append(
                CuratorProposal(existing_id, text.strip(), tuple(evidence_ids))
            )
        return tuple(parsed)


def _evidence_text(evidence: CurationEvidence) -> str:
    if evidence.correctness_success is True:
        correctness = "passed held-out correctness"
    elif evidence.correctness_success is False:
        correctness = "failed held-out correctness"
    else:
        correctness = "correctness not run"

    improvement = evidence.improvement_ratio
    if improvement is None:
        runtime = "trusted runtime change unavailable"
    elif improvement >= 0.0:
        runtime = f"trusted runtime improvement={improvement * 100.0:.2f}%"
    else:
        runtime = f"trusted runtime regression={abs(improvement) * 100.0:.2f}%"

    reflection = evidence.reflection
    return (
        f"- Evidence {evidence.candidate_id}; parent={evidence.parent_id}; "
        f"terminal_status={evidence.terminal_status}; "
        f"compile={'passed' if evidence.compile_success else 'failed'}; "
        f"{correctness}; {runtime}\n"
        f"  What changed: {reflection.what_changed}\n"
        f"  Observed: {reflection.observed}\n"
        f"  Post-hoc hypothesis: {reflection.hypothesis}\n"
        f"  Reflection confidence: {reflection.confidence}"
    )

"""Harness-controlled persistence and promotion for advisory LLM memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from kernelbunny.curator import CurationEvidence, CuratorProposal
from kernelbunny.models import MemoryLesson


@dataclass(frozen=True)
class MemoryUpdate:
    accepted_lesson_ids: tuple[str, ...]
    active_lessons: tuple[MemoryLesson, ...]


class MemoryBank:
    def __init__(self, root: Path, max_active_lessons: int) -> None:
        if max_active_lessons <= 0:
            raise ValueError("max_active_lessons must be positive.")
        self.root = root
        self.max_active_lessons = max_active_lessons
        self._lessons = self._load()

    def active_lessons(self) -> tuple[MemoryLesson, ...]:
        return tuple(lesson for lesson in self._lessons if lesson.status == "active")

    def curator_context(self) -> tuple[MemoryLesson, ...]:
        return tuple(
            lesson for lesson in self._lessons if lesson.status != "archived"
        )

    def apply(
        self,
        proposals: tuple[CuratorProposal, ...],
        evidence: tuple[CurationEvidence, ...],
        *,
        curation_id: str,
        large_improvement_ratio: float,
    ) -> MemoryUpdate:
        evidence_by_id = {item.candidate_id: item for item in evidence}
        lessons = {lesson.id: lesson for lesson in self._lessons}
        previous_status = {lesson.id: lesson.status for lesson in self._lessons}
        updated_ids: set[str] = set()
        accepted_ids: list[str] = []
        next_number = _next_lesson_number(self._lessons)

        for proposal in proposals:
            if proposal.existing_lesson_id is None:
                lesson_id = f"L{next_number:04d}"
                next_number += 1
                prior_evidence: tuple[str, ...] = ()
                was_active = False
            else:
                lesson_id = proposal.existing_lesson_id
                prior = lessons[lesson_id]
                prior_evidence = prior.evidence_candidate_ids
                was_active = prior.status == "active"

            evidence_ids = tuple(
                dict.fromkeys((*prior_evidence, *proposal.evidence_candidate_ids))
            )
            repeated = len(evidence_ids) >= 2
            large_improvement = any(
                (evidence_by_id[candidate_id].improvement_ratio or 0.0)
                >= large_improvement_ratio
                for candidate_id in proposal.evidence_candidate_ids
            )
            status = "active" if was_active or repeated or large_improvement else "low_priority"
            lessons[lesson_id] = MemoryLesson(
                lesson_id,
                proposal.lesson,
                evidence_ids,
                status,
            )
            updated_ids.add(lesson_id)
            accepted_ids.append(lesson_id)

        for lesson_id, lesson in tuple(lessons.items()):
            if previous_status.get(lesson_id) == "low_priority" and lesson_id not in updated_ids:
                lessons[lesson_id] = MemoryLesson(
                    lesson.id,
                    lesson.text,
                    lesson.evidence_candidate_ids,
                    "archived",
                )

        eligible = sorted(
            (lesson for lesson in lessons.values() if lesson.status == "active"),
            key=lambda lesson: (-len(lesson.evidence_candidate_ids), lesson.id),
        )
        for lesson in eligible[self.max_active_lessons :]:
            lessons[lesson.id] = MemoryLesson(
                lesson.id,
                lesson.text,
                lesson.evidence_candidate_ids,
                "low_priority",
            )

        self._lessons = tuple(sorted(lessons.values(), key=lambda lesson: lesson.id))
        self._persist(curation_id, proposals, tuple(accepted_ids))
        return MemoryUpdate(tuple(accepted_ids), self.active_lessons())

    def _load(self) -> tuple[MemoryLesson, ...]:
        path = self.root / "global_memory.json"
        if not path.is_file():
            return ()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"lessons"}:
            raise ValueError("Invalid global memory file.")
        return tuple(
            MemoryLesson(
                id=item["id"],
                text=item["text"],
                evidence_candidate_ids=tuple(item["evidence_candidate_ids"]),
                status=item["status"],
            )
            for item in payload["lessons"]
        )

    def _persist(
        self,
        curation_id: str,
        proposals: tuple[CuratorProposal, ...],
        accepted_ids: tuple[str, ...],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            self.root / "global_memory.json",
            {"lessons": [asdict(lesson) for lesson in self._lessons]},
        )
        active = self.active_lessons()
        markdown = "# KernelBunny Global Memory\n\n"
        if active:
            markdown += "\n".join(
                f"- **{lesson.id}**: {lesson.text} "
                f"(evidence: {', '.join(lesson.evidence_candidate_ids)})"
                for lesson in active
            )
            markdown += "\n"
        else:
            markdown += "No active lessons yet.\n"
        _write_text_atomic(self.root / "global_memory.md", markdown)

        archive_path = self.root / "curation_archive.jsonl"
        with archive_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "curation_id": curation_id,
                        "proposals": [asdict(proposal) for proposal in proposals],
                        "accepted_lesson_ids": list(accepted_ids),
                        "resulting_lessons": [asdict(lesson) for lesson in self._lessons],
                    }
                )
                + "\n"
            )


def _next_lesson_number(lessons: tuple[MemoryLesson, ...]) -> int:
    numbers = []
    for lesson in lessons:
        match = re.fullmatch(r"L(\d{4})", lesson.id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)

"""Core data records shared by the KernelBunny harness."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A CUDA source proposal and its search lineage."""

    id: str
    parent_id: str | None
    source_path: str
    worker_intent: str = ""
    revision: int = 0

    @property
    def revision_id(self) -> str:
        """Globally meaningful identity for this exact source revision."""

        return f"{self.id}/r{self.revision}"


@dataclass(frozen=True)
class Evaluation:
    """Trusted observations produced while evaluating one candidate."""

    candidate_id: str
    compile_success: bool
    compile_log: str
    correctness_success: bool | None
    correctness_log: str
    runtime_ms: float | None
    benchmark_log: str = ""


@dataclass(frozen=True)
class Reflection:
    """Untrusted post-hoc interpretation of one completed candidate lifecycle."""

    candidate_id: str
    what_changed: str
    observed: str
    hypothesis: str
    confidence: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("Reflection candidate_id cannot be empty.")
        for field_name in ("what_changed", "observed", "hypothesis"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"Reflection {field_name} cannot be empty.")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("Reflection confidence must be low, medium, or high.")


@dataclass(frozen=True)
class MemoryLesson:
    """Untrusted advice whose identity and visibility are harness-controlled."""

    id: str
    text: str
    evidence_candidate_ids: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Memory lesson id cannot be empty.")
        if not self.text.strip():
            raise ValueError("Memory lesson text cannot be empty.")
        if not self.evidence_candidate_ids:
            raise ValueError("Memory lessons require at least one evidence id.")
        if len(set(self.evidence_candidate_ids)) != len(self.evidence_candidate_ids):
            raise ValueError("Memory lesson evidence ids must be unique.")
        if self.status not in {"active", "low_priority", "archived"}:
            raise ValueError("Unknown memory lesson status.")

"""Trusted controller for candidate lifecycles and bounded iterative search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from kernelbunny.curator import CurationEvidence, Curator, CuratorError
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import ProviderError
from kernelbunny.memory import MemoryBank
from kernelbunny.models import Candidate, Evaluation, MemoryLesson, Reflection
from kernelbunny.reflector import (
    ReflectionError,
    Reflector,
    RevisionEvidence,
    render_reflection,
)
from kernelbunny.worker import ProposalError, Worker


CORRECTNESS_REPAIR_DIAGNOSTIC = (
    "The candidate failed a held-out correctness case. Review indexing, "
    "boundary handling, synchronization, initialization, and numerical "
    "behavior. Exact test inputs are hidden."
)


@dataclass(frozen=True)
class ControllerConfig:
    max_compile_repairs: int = 1
    max_correctness_repairs: int = 1
    population_size: int = 3
    children_per_parent: int = 2
    max_generations: int = 6
    max_candidates: int = 12
    max_no_improvement_generations: int = 3
    meaningful_improvement_ratio: float = 0.02
    curation_interval_generations: int = 3
    max_active_memory_lessons: int = 10
    max_curator_lessons_per_run: int = 5
    large_improvement_ratio: float = 0.10
    max_branch_reflections: int = 3

    def __post_init__(self) -> None:
        if self.max_compile_repairs < 0 or self.max_correctness_repairs < 0:
            raise ValueError("Repair budgets cannot be negative.")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.children_per_parent <= 0:
            raise ValueError("children_per_parent must be positive.")
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive.")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if self.max_no_improvement_generations <= 0:
            raise ValueError("max_no_improvement_generations must be positive.")
        if not 0.0 < self.meaningful_improvement_ratio < 1.0:
            raise ValueError("meaningful_improvement_ratio must be between 0 and 1.")
        if self.curation_interval_generations <= 0:
            raise ValueError("curation_interval_generations must be positive.")
        if self.max_active_memory_lessons <= 0:
            raise ValueError("max_active_memory_lessons must be positive.")
        if self.max_curator_lessons_per_run <= 0:
            raise ValueError("max_curator_lessons_per_run must be positive.")
        if not 0.0 < self.large_improvement_ratio < 1.0:
            raise ValueError("large_improvement_ratio must be between 0 and 1.")
        if self.max_branch_reflections <= 0:
            raise ValueError("max_branch_reflections must be positive.")


@dataclass(frozen=True)
class CandidateRunResult:
    candidate_id: str
    final_revision_id: str | None
    final_candidate: Candidate | None
    status: str
    evaluation: Evaluation | None
    budget_consumed: bool
    compile_repairs_used: int
    correctness_repairs_used: int
    reflection: Reflection | None
    reflection_status: str
    reflection_error: str


@dataclass(frozen=True)
class PopulationMember:
    candidate: Candidate
    evaluation: Evaluation

    def __post_init__(self) -> None:
        if self.evaluation.candidate_id != self.candidate.revision_id:
            raise ValueError("Population evaluation does not match its candidate.")
        if not _is_valid(self.evaluation):
            raise ValueError("Population members must have trusted valid evaluations.")


@dataclass(frozen=True)
class ChildAttempt:
    parent_revision_id: str
    child_number: int
    candidate_result: CandidateRunResult


@dataclass(frozen=True)
class GenerationStep:
    generation: int
    parent_revision_ids: tuple[str, ...]
    attempts: tuple[ChildAttempt, ...]
    meaningful_improvement: bool
    selected_revision_ids: tuple[str, ...]
    best_revision_id: str


@dataclass(frozen=True)
class CurationStep:
    generation: int
    status: str
    evidence_count: int
    accepted_lesson_ids: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class SearchResult:
    search_id: str
    stop_reason: str
    generations_attempted: int
    candidates_consumed: int
    no_improvement_generations: int
    active_population: tuple[PopulationMember, ...]
    best_candidate: Candidate
    best_evaluation: Evaluation
    steps: tuple[GenerationStep, ...]
    curations: tuple[CurationStep, ...]
    active_memory: tuple[MemoryLesson, ...]


class Controller:
    """Own candidate identity, repair authorization, and persistent state."""

    def __init__(
        self,
        worker: Worker,
        evaluator: Evaluator,
        runs_dir: Path,
        config: ControllerConfig | None = None,
        reflector: Reflector | None = None,
        curator: Curator | None = None,
        memory_dir: Path | None = None,
    ) -> None:
        self.worker = worker
        self.evaluator = evaluator
        self.runs_dir = runs_dir
        self.config = config or ControllerConfig()
        self.reflector = reflector
        self.curator = curator
        self.memory_bank = (
            MemoryBank(memory_dir, self.config.max_active_memory_lessons)
            if memory_dir is not None
            else None
        )

    def run_candidate(
        self,
        parent: Candidate,
        parent_evaluation: Evaluation,
        *,
        global_best_runtime_ms: float | None = None,
        branch_history: tuple[Reflection, ...] = (),
        global_memory: tuple[MemoryLesson, ...] = (),
    ) -> CandidateRunResult:
        if not _is_valid(parent_evaluation):
            raise ValueError("A candidate parent must have a trusted valid runtime.")
        if parent_evaluation.candidate_id != parent.revision_id:
            raise ValueError("Parent evaluation does not match the parent revision.")
        if global_best_runtime_ms is None:
            global_best_runtime_ms = parent_evaluation.runtime_ms
        if global_best_runtime_ms is None or global_best_runtime_ms <= 0.0:
            raise ValueError("Global-best runtime must be a positive trusted value.")

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        candidate_id, run_dir = _reserve_run_directory(self.runs_dir)
        compile_repairs = 0
        correctness_repairs = 0
        history: list[RevisionEvidence] = []

        initial_dir = _reserve_revision_directory(run_dir, 0)
        self._write_attempt_metadata(
            initial_dir,
            candidate_id=candidate_id,
            revision=0,
            parent_id=parent.revision_id,
            attempt_kind="initial",
            status="requesting_proposal",
        )
        self._write_root_metadata(
            run_dir,
            candidate_id=candidate_id,
            origin_parent_id=parent.revision_id,
            status="requesting_proposal",
            budget_consumed=False,
            final_revision_id=None,
            compile_repairs=compile_repairs,
            correctness_repairs=correctness_repairs,
        )

        try:
            raw_response = self.worker.request_proposal(
                parent,
                parent_evaluation,
                global_best_runtime_ms,
                branch_history=branch_history,
                global_memory=global_memory,
            )
        except ProviderError as exc:
            self._write_attempt_metadata(
                initial_dir,
                candidate_id=candidate_id,
                revision=0,
                parent_id=parent.revision_id,
                attempt_kind="initial",
                status="provider_failed",
                error=str(exc),
            )
            return self._finish(
                run_dir,
                candidate_id,
                parent,
                parent_evaluation,
                "provider_failed",
                None,
                None,
                False,
                compile_repairs,
                correctness_repairs,
                history,
            )

        _write_once(initial_dir / "raw_response.txt", raw_response)
        try:
            proposal = self.worker.parse_proposal(raw_response)
        except ProposalError as exc:
            self._write_attempt_metadata(
                initial_dir,
                candidate_id=candidate_id,
                revision=0,
                parent_id=parent.revision_id,
                attempt_kind="initial",
                status="proposal_failed",
                error=str(exc),
            )
            return self._finish(
                run_dir,
                candidate_id,
                parent,
                parent_evaluation,
                "proposal_failed",
                None,
                None,
                True,
                compile_repairs,
                correctness_repairs,
                history,
            )

        candidate = self._materialize_and_evaluate(
            initial_dir,
            candidate_id=candidate_id,
            revision=0,
            parent_id=parent.revision_id,
            attempt_kind="initial",
            worker_intent=proposal.worker_intent,
            cuda_source=proposal.cuda_source,
        )
        evaluation = self.evaluator.evaluate(candidate)
        self._archive_evaluation(initial_dir, candidate, evaluation, "initial")
        history.append(RevisionEvidence(candidate, evaluation, "initial"))

        revision = 0
        while True:
            terminal_status = self._terminal_status(
                evaluation,
                compile_repairs,
                correctness_repairs,
            )
            if terminal_status is not None:
                return self._finish(
                    run_dir,
                    candidate_id,
                    parent,
                    parent_evaluation,
                    terminal_status,
                    candidate,
                    evaluation,
                    True,
                    compile_repairs,
                    correctness_repairs,
                    history,
                )

            if not evaluation.compile_success:
                repair_kind = "compilation"
                diagnostic = evaluation.compile_log
                compile_repairs += 1
            else:
                repair_kind = "correctness"
                diagnostic = CORRECTNESS_REPAIR_DIAGNOSTIC
                correctness_repairs += 1

            revision += 1
            revision_dir = _reserve_revision_directory(run_dir, revision)
            self._write_attempt_metadata(
                revision_dir,
                candidate_id=candidate_id,
                revision=revision,
                parent_id=candidate.revision_id,
                attempt_kind=f"{repair_kind}_repair",
                status="requesting_repair",
            )
            self._write_root_metadata(
                run_dir,
                candidate_id=candidate_id,
                origin_parent_id=parent.revision_id,
                status="requesting_repair",
                budget_consumed=True,
                final_revision_id=candidate.revision_id,
                compile_repairs=compile_repairs,
                correctness_repairs=correctness_repairs,
            )

            try:
                raw_response = self.worker.request_repair(
                    original_parent=parent,
                    failed_candidate=candidate,
                    failure_kind=repair_kind,
                    diagnostic=diagnostic,
                )
            except ProviderError as exc:
                self._write_attempt_metadata(
                    revision_dir,
                    candidate_id=candidate_id,
                    revision=revision,
                    parent_id=candidate.revision_id,
                    attempt_kind=f"{repair_kind}_repair",
                    status="repair_provider_failed",
                    error=str(exc),
                )
                return self._finish(
                    run_dir,
                    candidate_id,
                    parent,
                    parent_evaluation,
                    "repair_provider_failed",
                    candidate,
                    evaluation,
                    True,
                    compile_repairs,
                    correctness_repairs,
                    history,
                )

            _write_once(revision_dir / "raw_response.txt", raw_response)
            try:
                proposal = self.worker.parse_proposal(raw_response)
            except ProposalError as exc:
                self._write_attempt_metadata(
                    revision_dir,
                    candidate_id=candidate_id,
                    revision=revision,
                    parent_id=candidate.revision_id,
                    attempt_kind=f"{repair_kind}_repair",
                    status="repair_proposal_failed",
                    error=str(exc),
                )
                return self._finish(
                    run_dir,
                    candidate_id,
                    parent,
                    parent_evaluation,
                    "repair_proposal_failed",
                    candidate,
                    evaluation,
                    True,
                    compile_repairs,
                    correctness_repairs,
                    history,
                )

            previous_revision_id = candidate.revision_id
            candidate = self._materialize_and_evaluate(
                revision_dir,
                candidate_id=candidate_id,
                revision=revision,
                parent_id=previous_revision_id,
                attempt_kind=f"{repair_kind}_repair",
                worker_intent=proposal.worker_intent,
                cuda_source=proposal.cuda_source,
            )
            evaluation = self.evaluator.evaluate(candidate)
            self._archive_evaluation(
                revision_dir,
                candidate,
                evaluation,
                f"{repair_kind}_repair",
            )
            history.append(
                RevisionEvidence(candidate, evaluation, f"{repair_kind}_repair")
            )

    def run_search(
        self,
        initial_candidate: Candidate,
        initial_evaluation: Evaluation,
    ) -> SearchResult:
        """Run a bounded evolutionary search from one trusted baseline."""

        if not _is_valid(initial_evaluation):
            raise ValueError("The initial search candidate must be trusted and valid.")
        if initial_evaluation.candidate_id != initial_candidate.revision_id:
            raise ValueError("Initial evaluation does not match the initial candidate.")

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        search_id, search_dir = _reserve_search_directory(self.runs_dir)
        active_population = [PopulationMember(initial_candidate, initial_evaluation)]
        best_candidate = initial_candidate
        best_evaluation = initial_evaluation
        generations = 0
        candidates_consumed = 0
        no_improvement = 0
        steps: list[GenerationStep] = []
        curations: list[CurationStep] = []
        pending_curation_evidence: list[CurationEvidence] = []
        branch_histories: dict[str, tuple[Reflection, ...]] = {
            initial_candidate.revision_id: ()
        }
        active_memory = (
            self.memory_bank.active_lessons() if self.memory_bank is not None else ()
        )

        self._write_search_state(
            search_dir,
            search_id=search_id,
            status="running",
            stop_reason=None,
            initial_candidate=initial_candidate,
            generations=generations,
            candidates_consumed=candidates_consumed,
            no_improvement=no_improvement,
            active_population=active_population,
            best_candidate=best_candidate,
            best_evaluation=best_evaluation,
            steps=steps,
            curations=curations,
            active_memory=active_memory,
        )

        while True:
            stop_reason = self._search_stop_reason(
                generations,
                candidates_consumed,
                no_improvement,
            )
            if stop_reason is not None:
                break

            generations += 1
            parents = list(active_population)
            attempts: list[ChildAttempt] = []
            offspring: list[tuple[PopulationMember, str]] = []
            meaningful_improvement = False
            generation_consumed = 0

            for child_number in range(1, self.config.children_per_parent + 1):
                for parent in parents:
                    if candidates_consumed >= self.config.max_candidates:
                        break
                    candidate_result = self.run_candidate(
                        parent.candidate,
                        parent.evaluation,
                        global_best_runtime_ms=best_evaluation.runtime_ms,
                        branch_history=branch_histories.get(
                            parent.candidate.revision_id,
                            (),
                        ),
                        global_memory=active_memory,
                    )
                    attempts.append(
                        ChildAttempt(
                            parent_revision_id=parent.candidate.revision_id,
                            child_number=child_number,
                            candidate_result=candidate_result,
                        )
                    )
                    final_candidate = candidate_result.final_candidate
                    if final_candidate is not None:
                        child_history = branch_histories.get(
                            parent.candidate.revision_id,
                            (),
                        )
                        if candidate_result.reflection is not None:
                            child_history = (*child_history, candidate_result.reflection)
                            child_history = child_history[
                                -self.config.max_branch_reflections :
                            ]
                            final_evaluation = candidate_result.evaluation
                            if final_evaluation is not None:
                                pending_curation_evidence.append(
                                    CurationEvidence(
                                        candidate_id=final_candidate.revision_id,
                                        parent_id=parent.candidate.revision_id,
                                        terminal_status=candidate_result.status,
                                        compile_success=final_evaluation.compile_success,
                                        correctness_success=(
                                            final_evaluation.correctness_success
                                        ),
                                        runtime_ms=final_evaluation.runtime_ms,
                                        parent_runtime_ms=parent.evaluation.runtime_ms,
                                        reflection=candidate_result.reflection,
                                    )
                                )
                        branch_histories[final_candidate.revision_id] = child_history
                    if not candidate_result.budget_consumed:
                        continue

                    candidates_consumed += 1
                    generation_consumed += 1
                    candidate = candidate_result.final_candidate
                    evaluation = candidate_result.evaluation
                    if (
                        candidate_result.status != "evaluated"
                        or candidate is None
                        or evaluation is None
                        or not _is_valid(evaluation)
                    ):
                        continue

                    member = PopulationMember(candidate, evaluation)
                    offspring.append((member, parent.candidate.revision_id))
                    if evaluation.runtime_ms < best_evaluation.runtime_ms:
                        best_candidate = candidate
                        best_evaluation = evaluation
                    if _is_meaningful_improvement(
                        parent.evaluation.runtime_ms,
                        evaluation.runtime_ms,
                        self.config.meaningful_improvement_ratio,
                    ):
                        meaningful_improvement = True
                if candidates_consumed >= self.config.max_candidates:
                    break

            selection_pool = [
                (member, member.candidate.revision_id) for member in parents
            ] + offspring
            active_population = _select_population(
                selection_pool,
                self.config.population_size,
            )

            if generation_consumed:
                if meaningful_improvement:
                    no_improvement = 0
                else:
                    no_improvement += 1

            step = GenerationStep(
                generation=generations,
                parent_revision_ids=tuple(
                    parent.candidate.revision_id for parent in parents
                ),
                attempts=tuple(attempts),
                meaningful_improvement=meaningful_improvement,
                selected_revision_ids=tuple(
                    member.candidate.revision_id for member in active_population
                ),
                best_revision_id=best_candidate.revision_id,
            )
            steps.append(step)

            if generations % self.config.curation_interval_generations == 0:
                curation = self._curate(
                    search_dir,
                    generation=generations,
                    curation_number=len(curations) + 1,
                    evidence=tuple(pending_curation_evidence),
                )
                curations.append(curation)
                if curation.status == "completed":
                    pending_curation_evidence.clear()
                    active_memory = (
                        self.memory_bank.active_lessons()
                        if self.memory_bank is not None
                        else ()
                    )

            next_stop_reason = self._search_stop_reason(
                generations,
                candidates_consumed,
                no_improvement,
            )
            self._write_search_state(
                search_dir,
                search_id=search_id,
                status="stopped" if next_stop_reason else "running",
                stop_reason=next_stop_reason,
                initial_candidate=initial_candidate,
                generations=generations,
                candidates_consumed=candidates_consumed,
                no_improvement=no_improvement,
                active_population=active_population,
                best_candidate=best_candidate,
                best_evaluation=best_evaluation,
                steps=steps,
                curations=curations,
                active_memory=active_memory,
            )

        return SearchResult(
            search_id=search_id,
            stop_reason=stop_reason,
            generations_attempted=generations,
            candidates_consumed=candidates_consumed,
            no_improvement_generations=no_improvement,
            active_population=tuple(active_population),
            best_candidate=best_candidate,
            best_evaluation=best_evaluation,
            steps=tuple(steps),
            curations=tuple(curations),
            active_memory=active_memory,
        )

    def _curate(
        self,
        search_dir: Path,
        *,
        generation: int,
        curation_number: int,
        evidence: tuple[CurationEvidence, ...],
    ) -> CurationStep:
        curation_dir = search_dir / "curations" / f"K{curation_number:04d}"
        curation_dir.mkdir(parents=True, exist_ok=False)
        if not evidence:
            step = CurationStep(generation, "no_evidence", 0, ())
            _write_json(curation_dir / "metadata.json", _curation_step_payload(step))
            return step
        if self.curator is None or self.memory_bank is None:
            step = CurationStep(generation, "not_configured", len(evidence), ())
            _write_json(curation_dir / "metadata.json", _curation_step_payload(step))
            return step

        existing_lessons = self.memory_bank.curator_context()
        try:
            raw_response = self.curator.request_curation(
                evidence,
                existing_lessons,
                self.config.max_curator_lessons_per_run,
            )
        except ProviderError as exc:
            error = str(exc)
            _write_once(curation_dir / "error.txt", error)
            step = CurationStep(
                generation,
                "provider_failed",
                len(evidence),
                (),
                error,
            )
            _write_json(curation_dir / "metadata.json", _curation_step_payload(step))
            return step

        _write_once(curation_dir / "raw_response.txt", raw_response)
        try:
            proposals = self.curator.parse_proposals(
                raw_response,
                allowed_evidence_ids={item.candidate_id for item in evidence},
                allowed_existing_lesson_ids={item.id for item in existing_lessons},
                max_lessons=self.config.max_curator_lessons_per_run,
            )
        except CuratorError as exc:
            error = str(exc)
            _write_once(curation_dir / "error.txt", error)
            step = CurationStep(
                generation,
                "parse_failed",
                len(evidence),
                (),
                error,
            )
            _write_json(curation_dir / "metadata.json", _curation_step_payload(step))
            return step

        _write_once(
            curation_dir / "proposals.json",
            json.dumps(
                {"lessons": [asdict(proposal) for proposal in proposals]},
                indent=2,
            )
            + "\n",
        )
        update = self.memory_bank.apply(
            proposals,
            evidence,
            curation_id=f"{search_dir.name}/K{curation_number:04d}",
            large_improvement_ratio=self.config.large_improvement_ratio,
        )
        step = CurationStep(
            generation,
            "completed",
            len(evidence),
            update.accepted_lesson_ids,
        )
        _write_json(curation_dir / "metadata.json", _curation_step_payload(step))
        return step

    def _search_stop_reason(
        self,
        generations: int,
        candidates_consumed: int,
        no_improvement: int,
    ) -> str | None:
        if candidates_consumed >= self.config.max_candidates:
            return "max_candidates"
        if no_improvement >= self.config.max_no_improvement_generations:
            return "no_meaningful_improvement"
        if generations >= self.config.max_generations:
            return "max_generations"
        return None

    def _terminal_status(
        self,
        evaluation: Evaluation,
        compile_repairs: int,
        correctness_repairs: int,
    ) -> str | None:
        if _is_valid(evaluation):
            return "evaluated"
        if not evaluation.compile_success:
            if compile_repairs >= self.config.max_compile_repairs:
                return "compile_repair_exhausted"
            return None
        if evaluation.correctness_success is False:
            if correctness_repairs >= self.config.max_correctness_repairs:
                return "correctness_repair_exhausted"
            return None
        if evaluation.correctness_success is True and evaluation.runtime_ms is None:
            return "benchmark_failed"
        return "evaluation_failed"

    def _materialize_and_evaluate(
        self,
        revision_dir: Path,
        *,
        candidate_id: str,
        revision: int,
        parent_id: str,
        attempt_kind: str,
        worker_intent: str,
        cuda_source: str,
    ) -> Candidate:
        source_path = revision_dir / "candidate.cu"
        _write_once(source_path, cuda_source)
        candidate = Candidate(
            id=candidate_id,
            parent_id=parent_id,
            source_path=str(source_path),
            worker_intent=worker_intent,
            revision=revision,
        )
        self._write_attempt_metadata(
            revision_dir,
            candidate_id=candidate_id,
            revision=revision,
            parent_id=parent_id,
            attempt_kind=attempt_kind,
            status="evaluating",
            worker_intent=worker_intent,
            source_path="candidate.cu",
        )
        return candidate

    def _archive_evaluation(
        self,
        revision_dir: Path,
        candidate: Candidate,
        evaluation: Evaluation,
        attempt_kind: str,
    ) -> None:
        _write_once(revision_dir / "compile.log", evaluation.compile_log)
        _write_once(revision_dir / "correctness.log", evaluation.correctness_log)
        _write_once(revision_dir / "benchmark.log", evaluation.benchmark_log)
        _write_once(
            revision_dir / "evaluation.json",
            json.dumps(asdict(evaluation), indent=2) + "\n",
        )
        self._write_attempt_metadata(
            revision_dir,
            candidate_id=candidate.id,
            revision=candidate.revision,
            parent_id=candidate.parent_id,
            attempt_kind=attempt_kind,
            status="evaluated",
            worker_intent=candidate.worker_intent,
            source_path="candidate.cu",
        )

    def _finish(
        self,
        run_dir: Path,
        candidate_id: str,
        original_parent: Candidate,
        parent_evaluation: Evaluation,
        status: str,
        final_candidate: Candidate | None,
        evaluation: Evaluation | None,
        budget_consumed: bool,
        compile_repairs: int,
        correctness_repairs: int,
        history: list[RevisionEvidence],
    ) -> CandidateRunResult:
        final_revision_id = evaluation.candidate_id if evaluation else None
        reflection, reflection_status, reflection_error = self._reflect(
            original_parent=original_parent,
            parent_evaluation=parent_evaluation,
            final_candidate=final_candidate,
            terminal_status=status,
            history=tuple(history),
        )
        self._write_root_metadata(
            run_dir,
            candidate_id=candidate_id,
            origin_parent_id=original_parent.revision_id,
            status=status,
            budget_consumed=budget_consumed,
            final_revision_id=final_revision_id,
            compile_repairs=compile_repairs,
            correctness_repairs=correctness_repairs,
            reflection_status=reflection_status,
        )
        return CandidateRunResult(
            candidate_id=candidate_id,
            final_revision_id=final_revision_id,
            final_candidate=final_candidate,
            status=status,
            evaluation=evaluation,
            budget_consumed=budget_consumed,
            compile_repairs_used=compile_repairs,
            correctness_repairs_used=correctness_repairs,
            reflection=reflection,
            reflection_status=reflection_status,
            reflection_error=reflection_error,
        )

    def _reflect(
        self,
        *,
        original_parent: Candidate,
        parent_evaluation: Evaluation,
        final_candidate: Candidate | None,
        terminal_status: str,
        history: tuple[RevisionEvidence, ...],
    ) -> tuple[Reflection | None, str, str]:
        if final_candidate is None or not history:
            return None, "not_applicable", ""
        if self.reflector is None:
            return None, "not_configured", ""

        revision_dir = Path(final_candidate.source_path).parent
        try:
            raw_response = self.reflector.request_reflection(
                original_parent=original_parent,
                parent_evaluation=parent_evaluation,
                final_candidate=final_candidate,
                terminal_status=terminal_status,
                history=history,
            )
        except ProviderError as exc:
            error = str(exc)
            _write_once(revision_dir / "reflection_error.txt", error)
            return None, "provider_failed", error

        _write_once(revision_dir / "reflection_raw_response.txt", raw_response)
        try:
            reflection = self.reflector.parse_reflection(
                raw_response,
                final_candidate.revision_id,
            )
        except ReflectionError as exc:
            error = str(exc)
            _write_once(revision_dir / "reflection_error.txt", error)
            return None, "parse_failed", error

        _write_once(
            revision_dir / "reflection.json",
            json.dumps(asdict(reflection), indent=2) + "\n",
        )
        _write_once(
            revision_dir / "reflection.md",
            render_reflection(reflection, final_candidate.worker_intent),
        )
        return reflection, "completed", ""

    def _write_search_state(
        self,
        search_dir: Path,
        *,
        search_id: str,
        status: str,
        stop_reason: str | None,
        initial_candidate: Candidate,
        generations: int,
        candidates_consumed: int,
        no_improvement: int,
        active_population: list[PopulationMember],
        best_candidate: Candidate,
        best_evaluation: Evaluation,
        steps: list[GenerationStep],
        curations: list[CurationStep],
        active_memory: tuple[MemoryLesson, ...],
    ) -> None:
        _write_json(
            search_dir / "state.json",
            {
                "search_id": search_id,
                "status": status,
                "stop_reason": stop_reason,
                "config": asdict(self.config),
                "initial_revision_id": initial_candidate.revision_id,
                "generations_attempted": generations,
                "candidates_consumed": candidates_consumed,
                "no_improvement_generations": no_improvement,
                "active_population": [
                    {
                        "revision_id": member.candidate.revision_id,
                        "runtime_ms": member.evaluation.runtime_ms,
                    }
                    for member in active_population
                ],
                "best_ever": {
                    "revision_id": best_candidate.revision_id,
                    "runtime_ms": best_evaluation.runtime_ms,
                },
                "generations": [_generation_step_payload(step) for step in steps],
                "curations": [_curation_step_payload(step) for step in curations],
                "active_memory": [asdict(lesson) for lesson in active_memory],
            },
        )

    @staticmethod
    def _write_attempt_metadata(
        revision_dir: Path,
        *,
        candidate_id: str,
        revision: int,
        parent_id: str | None,
        attempt_kind: str,
        status: str,
        worker_intent: str = "",
        source_path: str | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "candidate_id": candidate_id,
            "revision_id": f"{candidate_id}/r{revision}",
            "parent_id": parent_id,
            "attempt_kind": attempt_kind,
            "status": status,
        }
        if worker_intent:
            payload["worker_intent"] = worker_intent
        if source_path:
            payload["source_path"] = source_path
        if error:
            payload["error"] = error
        _write_json(revision_dir / "metadata.json", payload)

    @staticmethod
    def _write_root_metadata(
        run_dir: Path,
        *,
        candidate_id: str,
        origin_parent_id: str,
        status: str,
        budget_consumed: bool,
        final_revision_id: str | None,
        compile_repairs: int,
        correctness_repairs: int,
        reflection_status: str = "pending",
    ) -> None:
        _write_json(
            run_dir / "metadata.json",
            {
                "id": candidate_id,
                "origin_parent_id": origin_parent_id,
                "status": status,
                "budget_consumed": budget_consumed,
                "final_revision_id": final_revision_id,
                "compile_repairs_used": compile_repairs,
                "correctness_repairs_used": correctness_repairs,
                "reflection_status": reflection_status,
            },
        )


def _reserve_run_directory(runs_dir: Path) -> tuple[str, Path]:
    used_numbers = []
    for path in runs_dir.iterdir():
        match = re.fullmatch(r"C(\d{4})", path.name)
        if match:
            used_numbers.append(int(match.group(1)))
    candidate_id = f"C{max(used_numbers, default=0) + 1:04d}"
    run_dir = runs_dir / candidate_id
    run_dir.mkdir(exist_ok=False)
    return candidate_id, run_dir


def _reserve_revision_directory(run_dir: Path, revision: int) -> Path:
    revision_dir = run_dir / f"r{revision}"
    revision_dir.mkdir(exist_ok=False)
    return revision_dir


def _reserve_search_directory(runs_dir: Path) -> tuple[str, Path]:
    searches_dir = runs_dir / "searches"
    searches_dir.mkdir(exist_ok=True)
    used_numbers = []
    for path in searches_dir.iterdir():
        match = re.fullmatch(r"S(\d{4})", path.name)
        if match:
            used_numbers.append(int(match.group(1)))
    search_id = f"S{max(used_numbers, default=0) + 1:04d}"
    search_dir = searches_dir / search_id
    search_dir.mkdir(exist_ok=False)
    return search_id, search_dir


def _is_valid(evaluation: Evaluation) -> bool:
    return (
        evaluation.compile_success
        and evaluation.correctness_success is True
        and evaluation.runtime_ms is not None
    )


def _is_meaningful_improvement(
    parent_runtime_ms: float | None,
    candidate_runtime_ms: float | None,
    improvement_ratio: float,
) -> bool:
    if parent_runtime_ms is None or candidate_runtime_ms is None:
        return False
    return candidate_runtime_ms <= parent_runtime_ms * (1.0 - improvement_ratio)


def _select_population(
    entries: list[tuple[PopulationMember, str]],
    population_size: int,
) -> list[PopulationMember]:
    """Select two fast members, then preserve one unrepresented parent branch."""

    ranked = sorted(
        entries,
        key=lambda entry: (
            entry[0].evaluation.runtime_ms,
            entry[0].candidate.revision_id,
        ),
    )
    selected = ranked[: min(2, population_size, len(ranked))]

    if len(selected) < population_size:
        represented_families = {family_id for _, family_id in selected}
        exploration = next(
            (
                entry
                for entry in ranked
                if entry not in selected and entry[1] not in represented_families
            ),
            None,
        )
        if exploration is not None:
            selected.append(exploration)

    for entry in ranked:
        if len(selected) >= population_size:
            break
        if entry not in selected:
            selected.append(entry)

    return [member for member, _ in selected]


def _generation_step_payload(step: GenerationStep) -> dict[str, object]:
    return {
        "generation": step.generation,
        "parent_revision_ids": list(step.parent_revision_ids),
        "attempts": [_child_attempt_payload(attempt) for attempt in step.attempts],
        "meaningful_improvement": step.meaningful_improvement,
        "selected_revision_ids": list(step.selected_revision_ids),
        "best_revision_id": step.best_revision_id,
    }


def _child_attempt_payload(attempt: ChildAttempt) -> dict[str, object]:
    result = attempt.candidate_result
    evaluation = result.evaluation
    return {
        "parent_revision_id": attempt.parent_revision_id,
        "child_number": attempt.child_number,
        "candidate_id": result.candidate_id,
        "final_revision_id": result.final_revision_id,
        "status": result.status,
        "budget_consumed": result.budget_consumed,
        "compile_repairs_used": result.compile_repairs_used,
        "correctness_repairs_used": result.correctness_repairs_used,
        "reflection_status": result.reflection_status,
        "runtime_ms": evaluation.runtime_ms if evaluation else None,
    }


def _curation_step_payload(step: CurationStep) -> dict[str, object]:
    return {
        "generation": step.generation,
        "status": step.status,
        "evidence_count": step.evidence_count,
        "accepted_lesson_ids": list(step.accepted_lesson_ids),
        "error": step.error,
    }


def _write_once(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)

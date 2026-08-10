"""Opt-in Phase 9 acceptance test using the real CLI, DeepSeek, and CUDA."""

from __future__ import annotations

import json
from pathlib import Path
import re

from kernelbunny.main import main as kernelbunny_main


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_ROOT = PROJECT_ROOT / "runs" / "phase9-live"


def _reserve_session() -> Path:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in LIVE_ROOT.iterdir():
        match = re.fullmatch(r"T(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    session = LIVE_ROOT / f"T{max(numbers, default=0) + 1:04d}"
    session.mkdir(exist_ok=False)
    return session


def main() -> int:
    session = _reserve_session()
    runs_dir = session / "candidates"
    memory_dir = session / "memory"
    exit_code = kernelbunny_main(
        [
            "search",
            "--runs-dir",
            str(runs_dir),
            "--memory-dir",
            str(memory_dir),
            "--population-size",
            "2",
            "--children-per-parent",
            "1",
            "--max-generations",
            "2",
            "--max-candidates",
            "3",
            "--max-no-improvement-generations",
            "2",
            "--curation-interval-generations",
            "1",
            "--log-level",
            "INFO",
        ]
    )

    state_paths = sorted((runs_dir / "searches").glob("S*/state.json"))
    if not state_paths:
        print(f"session={session}")
        print("No search state was written.")
        return 1
    state_path = state_paths[-1]
    state = json.loads(state_path.read_text(encoding="utf-8"))

    candidate_summaries = []
    for metadata_path in sorted(runs_dir.glob("C*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        final_revision_id = metadata.get("final_revision_id")
        evaluation = None
        if isinstance(final_revision_id, str) and "/r" in final_revision_id:
            revision = final_revision_id.rsplit("/r", 1)[1]
            evaluation_path = metadata_path.parent / f"r{revision}" / "evaluation.json"
            if evaluation_path.is_file():
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        candidate_summaries.append(
            {
                "candidate_id": metadata.get("id"),
                "final_revision_id": final_revision_id,
                "status": metadata.get("status"),
                "compile_repairs": metadata.get("compile_repairs_used"),
                "correctness_repairs": metadata.get("correctness_repairs_used"),
                "runtime_ms": evaluation.get("runtime_ms") if evaluation else None,
                "correct": evaluation.get("correctness_success") if evaluation else None,
            }
        )

    print(f"session={session}")
    print(f"search_state={state_path}")
    print(
        f"stop_reason={state['stop_reason']} "
        f"generations={state['generations_attempted']} "
        f"candidates={state['candidates_consumed']}"
    )
    print(f"best_ever={state['best_ever']}")
    for summary in candidate_summaries:
        print(f"candidate={summary}")
    for curation in state["curations"]:
        print(f"curation={curation}")

    curation_metadata = list(
        (runs_dir / "searches").glob("S*/curations/K*/metadata.json")
    )
    checks = (
        exit_code == 0,
        state["status"] == "stopped",
        state["generations_attempted"] == 2,
        1 <= state["candidates_consumed"] <= 3,
        len(state["generations"]) == 2,
        len(state["curations"]) == 2,
        len(curation_metadata) == 2,
        any(runs_dir.glob("C*/r*/raw_response.txt")),
        bool(state["active_population"]),
        isinstance(state["best_ever"]["runtime_ms"], float),
    )
    if not all(checks):
        print(f"checks={checks}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

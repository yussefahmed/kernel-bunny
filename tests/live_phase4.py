"""Opt-in Phase 4 test using the real DeepSeek API and CUDA evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from kernelbunny.controller import Controller, ControllerConfig
from kernelbunny.evaluator import Evaluator
from kernelbunny.llm import DeepSeekLLM, load_env_file
from kernelbunny.models import Candidate
from kernelbunny.worker import Worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "kernels" / "baseline.cu"
LIVE_RUNS = PROJECT_ROOT / "runs" / "phase4-live"

COMPILE_FAILURE = r'''
#include <cuda_runtime.h>
__global__ void broken_kernel(float* c) {
    float value = ;
    if (threadIdx.x == 0) c[0] = value;
}
extern "C" void launch_sgemm(
    const float*, const float*, float* c, int, int, int, cudaStream_t stream) {
    broken_kernel<<<1, 1, 0, stream>>>(c);
}
'''.strip()

CORRECTNESS_FAILURE = r'''
#include <cuda_runtime.h>
__global__ void zero_kernel(float* c, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < m * n) c[index] = 0.0f;
}
extern "C" void launch_sgemm(
    const float*, const float*, float* c, int m, int n, int, cudaStream_t stream) {
    int count = m * n;
    zero_kernel<<<(count + 255) / 256, 256, 0, stream>>>(c, m, n);
}
'''.strip()


class SeededWorker:
    """Seed a known failure, then delegate repairs to the real API worker."""

    def __init__(self, real_worker: Worker, source: str, intent: str) -> None:
        self.real_worker = real_worker
        self.initial_response = json.dumps(
            {"worker_intent": intent, "cuda_source": source}
        )

    def request_proposal(
        self, parent, parent_evaluation, global_best_runtime_ms, **kwargs
    ):
        return self.initial_response

    def request_repair(
        self,
        original_parent,
        failed_candidate,
        failure_kind,
        diagnostic,
    ):
        return self.real_worker.request_repair(
            original_parent,
            failed_candidate,
            failure_kind,
            diagnostic,
        )

    def parse_proposal(self, raw_response):
        return self.real_worker.parse_proposal(raw_response)


def run_scenario(
    name: str,
    source: str,
    config: ControllerConfig,
    baseline: Candidate,
    baseline_evaluation,
    evaluator: Evaluator,
) -> bool:
    worker = SeededWorker(
        Worker(DeepSeekLLM()),
        source,
        f"Controlled live {name} failure for Phase 4 verification.",
    )
    controller = Controller(
        worker=worker,
        evaluator=evaluator,
        runs_dir=LIVE_RUNS / name,
        config=config,
    )
    result = controller.run_candidate(baseline, baseline_evaluation)
    runtime = result.evaluation.runtime_ms if result.evaluation else None
    print(
        f"{name}: id={result.candidate_id}, status={result.status}, "
        f"revision={result.final_revision_id}, runtime_ms={runtime}, "
        f"compile_repairs={result.compile_repairs_used}, "
        f"correctness_repairs={result.correctness_repairs_used}"
    )
    return result.status == "evaluated"


def main() -> int:
    load_env_file(PROJECT_ROOT / ".env")
    evaluator = Evaluator()
    baseline = Candidate(
        "baseline",
        None,
        str(BASELINE_PATH),
        "Trusted naive starting kernel.",
    )
    baseline_evaluation = evaluator.evaluate(baseline)
    if baseline_evaluation.runtime_ms is None:
        print("Live Phase 4 tests require a working CUDA baseline.")
        return 1

    print(f"live baseline runtime_ms={baseline_evaluation.runtime_ms}")
    compile_ok = run_scenario(
        "compile",
        COMPILE_FAILURE,
        ControllerConfig(max_compile_repairs=1, max_correctness_repairs=0),
        baseline,
        baseline_evaluation,
        evaluator,
    )
    correctness_ok = run_scenario(
        "correctness",
        CORRECTNESS_FAILURE,
        ControllerConfig(max_compile_repairs=0, max_correctness_repairs=1),
        baseline,
        baseline_evaluation,
        evaluator,
    )
    return 0 if compile_ok and correctness_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>


extern "C" void launch_sgemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream
);


namespace {

constexpr float kAbsoluteTolerance = 1.0e-3f;
constexpr float kRelativeTolerance = 1.0e-3f;


struct Problem {
    int m;
    int n;
    int k;
    int seed;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> expected;
};


struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    ~DeviceBuffers() {
        if (a != nullptr) {
            cudaFree(a);
        }
        if (b != nullptr) {
            cudaFree(b);
        }
        if (c != nullptr) {
            cudaFree(c);
        }
    }
};


struct Validation {
    bool success;
    std::size_t mismatch_count;
    float max_absolute_error;
    float max_relative_error;
    std::size_t first_mismatch;
    float first_expected;
    float first_actual;
};


bool check_cuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cout << "KB_STATUS=ERROR\n";
    std::cout << "KB_ERROR=" << operation << ": " << cudaGetErrorString(result) << "\n";
    return false;
}


Problem make_problem(int m, int n, int k, int seed) {
    Problem problem{
        m,
        n,
        k,
        seed,
        std::vector<float>(static_cast<std::size_t>(m) * k),
        std::vector<float>(static_cast<std::size_t>(k) * n),
        std::vector<float>(static_cast<std::size_t>(m) * n),
    };

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& value : problem.a) {
        value = distribution(generator);
    }
    for (float& value : problem.b) {
        value = distribution(generator);
    }

    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += problem.a[static_cast<std::size_t>(row) * k + inner]
                    * problem.b[static_cast<std::size_t>(inner) * n + column];
            }
            problem.expected[static_cast<std::size_t>(row) * n + column] = sum;
        }
    }
    return problem;
}


bool initialize_device(const Problem& problem, DeviceBuffers& device) {
    const std::size_t a_bytes = problem.a.size() * sizeof(float);
    const std::size_t b_bytes = problem.b.size() * sizeof(float);
    const std::size_t c_bytes = problem.expected.size() * sizeof(float);

    return check_cuda(cudaMalloc(&device.a, a_bytes), "cudaMalloc(A)")
        && check_cuda(cudaMalloc(&device.b, b_bytes), "cudaMalloc(B)")
        && check_cuda(cudaMalloc(&device.c, c_bytes), "cudaMalloc(C)")
        && check_cuda(
            cudaMemcpy(device.a, problem.a.data(), a_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(A)"
        )
        && check_cuda(
            cudaMemcpy(device.b, problem.b.data(), b_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(B)"
        );
}


bool launch_once(
    const Problem& problem,
    DeviceBuffers& device,
    double& elapsed_ms
) {
    const std::size_t c_bytes = problem.expected.size() * sizeof(float);
    if (!check_cuda(cudaMemset(device.c, 0xFF, c_bytes), "cudaMemset(C)")) {
        return false;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "pre-launch synchronization")) {
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    launch_sgemm(
        device.a,
        device.b,
        device.c,
        problem.m,
        problem.n,
        problem.k,
        nullptr
    );
    if (!check_cuda(cudaGetLastError(), "candidate launch")) {
        return false;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "candidate synchronization")) {
        return false;
    }
    const auto end = std::chrono::steady_clock::now();

    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return true;
}


Validation validate_output(const Problem& problem, const DeviceBuffers& device) {
    std::vector<float> actual(problem.expected.size());
    const std::size_t c_bytes = actual.size() * sizeof(float);
    if (!check_cuda(
            cudaMemcpy(actual.data(), device.c, c_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(C)"
        )) {
        return {false, actual.size(), INFINITY, INFINITY, 0, 0.0f, 0.0f};
    }

    Validation result{true, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f};
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float expected = problem.expected[index];
        const float observed = actual[index];
        const float absolute_error = std::fabs(observed - expected);
        const float denominator = std::max(std::fabs(expected), kAbsoluteTolerance);
        const float relative_error = absolute_error / denominator;
        const float allowed_error = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);

        result.max_absolute_error = std::max(result.max_absolute_error, absolute_error);
        result.max_relative_error = std::max(result.max_relative_error, relative_error);
        if (!std::isfinite(observed) || absolute_error > allowed_error) {
            if (result.mismatch_count == 0) {
                result.first_mismatch = index;
                result.first_expected = expected;
                result.first_actual = observed;
            }
            ++result.mismatch_count;
            result.success = false;
        }
    }
    return result;
}


void print_validation(const Validation& validation) {
    std::cout << std::setprecision(9);
    std::cout << "KB_MISMATCH_COUNT=" << validation.mismatch_count << "\n";
    std::cout << "KB_MAX_ABS_ERROR=" << validation.max_absolute_error << "\n";
    std::cout << "KB_MAX_REL_ERROR=" << validation.max_relative_error << "\n";
    if (!validation.success) {
        std::cout << "KB_FIRST_MISMATCH_INDEX=" << validation.first_mismatch << "\n";
        std::cout << "KB_FIRST_EXPECTED=" << validation.first_expected << "\n";
        std::cout << "KB_FIRST_ACTUAL=" << validation.first_actual << "\n";
    }
}


int run_correctness(const Problem& problem) {
    DeviceBuffers device;
    if (!initialize_device(problem, device)) {
        return 2;
    }

    double ignored_time = 0.0;
    if (!launch_once(problem, device, ignored_time)) {
        return 2;
    }

    const Validation validation = validate_output(problem, device);
    std::cout << "KB_STATUS=" << (validation.success ? "PASS" : "FAIL") << "\n";
    print_validation(validation);
    return validation.success ? 0 : 1;
}


int run_benchmark(const Problem& problem, int warmup_runs, int measured_runs) {
    if (warmup_runs < 0 || measured_runs <= 0) {
        std::cout << "KB_STATUS=ERROR\n";
        std::cout << "KB_ERROR=Warm-up count must be non-negative and measured count positive.\n";
        return 2;
    }

    DeviceBuffers device;
    if (!initialize_device(problem, device)) {
        return 2;
    }

    double ignored_time = 0.0;
    for (int run = 0; run < warmup_runs; ++run) {
        if (!launch_once(problem, device, ignored_time)) {
            return 2;
        }
    }

    std::vector<double> samples;
    samples.reserve(measured_runs);
    for (int run = 0; run < measured_runs; ++run) {
        double elapsed_ms = 0.0;
        if (!launch_once(problem, device, elapsed_ms)) {
            return 2;
        }

        const Validation validation = validate_output(problem, device);
        if (!validation.success) {
            std::cout << "KB_STATUS=FAIL\n";
            std::cout << "KB_ERROR=Candidate produced incorrect output during benchmarking.\n";
            print_validation(validation);
            return 1;
        }
        samples.push_back(elapsed_ms);
    }

    std::cout << "KB_STATUS=PASS\n";
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "KB_TIMES_MS=";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << samples[index];
    }
    std::cout << "\n";
    return 0;
}


int parse_positive_integer(const char* text, const char* name) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

}  // namespace


int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 8) {
            std::cout << "KB_STATUS=ERROR\n";
            std::cout << "KB_ERROR=Invalid runner arguments.\n";
            return 2;
        }

        const std::string mode = argv[1];
        const int m = parse_positive_integer(argv[2], "M");
        const int n = parse_positive_integer(argv[3], "N");
        const int k = parse_positive_integer(argv[4], "K");
        const int seed = std::stoi(argv[5]);
        const Problem problem = make_problem(m, n, k, seed);

        if (mode == "correctness" && argc == 6) {
            return run_correctness(problem);
        }
        if (mode == "benchmark" && argc == 8) {
            const int warmup_runs = std::stoi(argv[6]);
            const int measured_runs = parse_positive_integer(argv[7], "measured runs");
            return run_benchmark(problem, warmup_runs, measured_runs);
        }

        std::cout << "KB_STATUS=ERROR\n";
        std::cout << "KB_ERROR=Mode and argument count do not match.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cout << "KB_STATUS=ERROR\n";
        std::cout << "KB_ERROR=" << error.what() << "\n";
        return 2;
    }
}

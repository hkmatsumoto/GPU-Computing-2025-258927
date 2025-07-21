#include "utils/timer.h"
#include "utils/cuda_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// CUDA Timer implementation
CudaTimer::CudaTimer() : started(false) {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void CudaTimer::start() {
    CUDA_CHECK(cudaEventRecord(start_event));
    started = true;
}

void CudaTimer::stop() {
    if (!started) {
        std::cerr << "Warning: Timer not started before stop" << std::endl;
        return;
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    started = false;
}

float CudaTimer::get_elapsed_time() {
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
    return time_ms;
}

// CPU Timer implementation
CPUTimer::CPUTimer() : started(false) {}

void CPUTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    started = true;
}

void CPUTimer::stop() {
    if (!started) {
        std::cerr << "Warning: Timer not started before stop" << std::endl;
        return;
    }
    end_time = std::chrono::high_resolution_clock::now();
    started = false;
}

double CPUTimer::get_elapsed_time() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// Benchmark Runner implementation
BenchmarkRunner::BenchmarkRunner(int warmup, int benchmark) 
    : warmup_iterations(warmup), benchmark_iterations(benchmark) {}

void BenchmarkRunner::add_time(float time_ms) {
    times.push_back(time_ms);
}

void BenchmarkRunner::reset() {
    times.clear();
}

double BenchmarkRunner::get_mean_time() {
    if (times.empty()) return 0.0;
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

double BenchmarkRunner::get_geometric_mean_time() {
    if (times.empty()) return 0.0;
    
    double log_sum = 0.0;
    for (float time : times) {
        if (time > 0) {
            log_sum += std::log(time);
        }
    }
    return std::exp(log_sum / times.size());
}

double BenchmarkRunner::get_std_deviation() {
    if (times.size() < 2) return 0.0;
    
    double mean = get_mean_time();
    double variance = 0.0;
    
    for (float time : times) {
        double diff = time - mean;
        variance += diff * diff;
    }
    
    variance /= (times.size() - 1);
    return std::sqrt(variance);
}

double BenchmarkRunner::get_min_time() {
    if (times.empty()) return 0.0;
    return *std::min_element(times.begin(), times.end());
}

double BenchmarkRunner::get_max_time() {
    if (times.empty()) return 0.0;
    return *std::max_element(times.begin(), times.end());
}

void BenchmarkRunner::print_statistics() {
    if (times.empty()) {
        std::cout << "No timing data available" << std::endl;
        return;
    }
    
    std::cout << "Benchmark Statistics:" << std::endl;
    std::cout << "  Iterations: " << times.size() << std::endl;
    std::cout << "  Mean time: " << get_mean_time() << " ms" << std::endl;
    std::cout << "  Geometric mean: " << get_geometric_mean_time() << " ms" << std::endl;
    std::cout << "  Std deviation: " << get_std_deviation() << " ms" << std::endl;
    std::cout << "  Min time: " << get_min_time() << " ms" << std::endl;
    std::cout << "  Max time: " << get_max_time() << " ms" << std::endl;
} 
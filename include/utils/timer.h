#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// CUDA event-based timer for GPU operations
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool started;

public:
    CudaTimer();
    ~CudaTimer();
    
    void start();
    void stop();
    float get_elapsed_time(); // returns time in milliseconds
};

// CPU timer for host operations
class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool started;

public:
    CPUTimer();
    
    void start();
    void stop();
    double get_elapsed_time(); // returns time in milliseconds
};

// Benchmark runner for multiple iterations
class BenchmarkRunner {
private:
    std::vector<float> times;
    int warmup_iterations;
    int benchmark_iterations;

public:
    BenchmarkRunner(int warmup = 3, int benchmark = 10);
    
    void add_time(float time_ms);
    void reset();
    
    double get_mean_time();
    double get_geometric_mean_time();
    double get_std_deviation();
    double get_min_time();
    double get_max_time();
    
    void print_statistics();
};

#endif // TIMER_H 
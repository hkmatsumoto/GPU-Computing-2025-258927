#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "utils/matrix_io.h"
#include "utils/timer.h"
#include <string>
#include <vector>

// Performance metrics structure
struct PerformanceMetrics {
    double execution_time_ms;
    double memory_bandwidth_gb_s;
    double flops;
    double shared_memory_usage_kb;
    std::string kernel_name;
    bool is_result_correct;
    double max_error;
};

// Benchmark configuration
struct BenchmarkConfig {
    int warmup_iterations = 3;
    int benchmark_iterations = 10;
    bool verify_results = true;
    bool profile_memory = true;
    std::vector<std::string> matrix_files;
};

// SpMV kernel function pointer type
typedef void (*SpMVKernel)(int num_rows, const double* values, 
                          const int* col_indices, const int* row_offsets,
                          const double* x, double* y);

// Benchmark a single SpMV kernel
PerformanceMetrics benchmark_spmv_kernel(const CSRMatrix& matrix, 
                                          SpMVKernel kernel,
                                          const std::string& kernel_name,
                                          const BenchmarkConfig& config);

// Benchmark cuSPARSE implementation
PerformanceMetrics benchmark_cusparse_spmv(const CSRMatrix& matrix,
                                            const BenchmarkConfig& config);

// Run comprehensive benchmark comparing all kernels
void run_comprehensive_benchmark(const std::vector<std::string>& matrix_files,
                                 const BenchmarkConfig& config);

// Verify SpMV result correctness
bool verify_spmv_result(const CSRMatrix& matrix, const double* x, 
                        const double* y_gpu, const double* y_reference,
                        double tolerance = 1e-12);

// Generate reference result using CPU
void compute_spmv_reference(const CSRMatrix& matrix, const double* x, double* y);

// Print performance comparison table
void print_performance_table(const std::vector<PerformanceMetrics>& results);

// Save results to CSV file
void save_results_to_csv(const std::vector<PerformanceMetrics>& results,
                         const std::string& filename);

#endif // BENCHMARK_H 
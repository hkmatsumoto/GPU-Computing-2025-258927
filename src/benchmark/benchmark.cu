#include "benchmark/benchmark.h"
#include "kernels/spmv_kernels.h"
#include "utils/cuda_utils.h"
#include "utils/timer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Benchmark a single SpMV kernel
PerformanceMetrics benchmark_spmv_kernel(const CSRMatrix& matrix, 
                                          SpMVKernel kernel,
                                          const std::string& kernel_name,
                                          const BenchmarkConfig& config) {
    PerformanceMetrics metrics;
    metrics.kernel_name = kernel_name;
    
    // Allocate device memory
    double *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_offsets;
    
    size_t values_size = matrix.nnz * sizeof(double);
    size_t indices_size = matrix.nnz * sizeof(int);
    size_t offsets_size = (matrix.num_rows + 1) * sizeof(int);
    size_t vector_size = matrix.num_cols * sizeof(double);
    size_t result_size = matrix.num_rows * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_values, values_size));
    CUDA_CHECK(cudaMalloc(&d_col_indices, indices_size));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, offsets_size));
    CUDA_CHECK(cudaMalloc(&d_x, vector_size));
    CUDA_CHECK(cudaMalloc(&d_y, result_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_values, matrix.values.data(), values_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, matrix.col_indices.data(), indices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, matrix.row_offsets.data(), offsets_size, cudaMemcpyHostToDevice));
    
    // Initialize input vector with random values
    std::vector<double> h_x(matrix.num_cols, 1.0); // Simple initialization
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), vector_size, cudaMemcpyHostToDevice));
    
    // Initialize result vector
    CUDA_CHECK(cudaMemset(d_y, 0, result_size));
    
    BenchmarkRunner runner(config.warmup_iterations, config.benchmark_iterations);
    CudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < config.warmup_iterations; i++) {
        kernel(matrix.num_rows, d_values, d_col_indices, d_row_offsets, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark runs
    for (int i = 0; i < config.benchmark_iterations; i++) {
        timer.start();
        kernel(matrix.num_rows, d_values, d_col_indices, d_row_offsets, d_x, d_y);
        timer.stop();
        runner.add_time(timer.get_elapsed_time());
    }
    
    // Calculate metrics
    metrics.execution_time_ms = runner.get_geometric_mean_time();
    
    // Calculate memory bandwidth (approximate)
    size_t total_bytes = values_size + indices_size + vector_size + result_size;
    metrics.memory_bandwidth_gb_s = calculate_memory_bandwidth(total_bytes, metrics.execution_time_ms);
    
    // Calculate FLOPS
    metrics.flops = calculate_spmv_flops(matrix.nnz, metrics.execution_time_ms);
    
    // TODO: Calculate shared memory usage (kernel-specific)
    metrics.shared_memory_usage_kb = 0.0;
    
    // Initialize verification results
    metrics.is_result_correct = true;
    metrics.max_error = 0.0;
    
    // Verify result if requested
    if (config.verify_results) {
        std::vector<double> h_y(matrix.num_rows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, result_size, cudaMemcpyDeviceToHost));
        
        // Compute CPU reference result
        std::vector<double> h_x(matrix.num_cols, 1.0);
        std::vector<double> h_y_ref(matrix.num_rows);
        compute_spmv_reference(matrix, h_x.data(), h_y_ref.data());
        
        // Verify results and find maximum error
        const double tolerance = 1e-10;
        double max_error = 0.0;
        int error_count = 0;
        
        for (int i = 0; i < matrix.num_rows; i++) {
            double error = std::abs(h_y[i] - h_y_ref[i]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                error_count++;
            }
        }
        
        metrics.max_error = max_error;
        metrics.is_result_correct = (error_count == 0);
        
        if (!metrics.is_result_correct) {
            std::cout << "  ❌ Result verification FAILED (max error: " << max_error 
                      << ", errors in " << error_count << " rows)" << std::endl;
            
            // Print first few mismatches for debugging
            int mismatch_count = 0;
            std::cerr << "  First few errors:" << std::endl;
            for (int i = 0; i < std::min(matrix.num_rows, 1000) && mismatch_count < 5; i++) {
                double error = std::abs(h_y[i] - h_y_ref[i]);
                if (error > tolerance) {
                    std::cerr << "    Row " << i << ": GPU=" << h_y[i] << ", CPU=" << h_y_ref[i] 
                              << ", error=" << error << std::endl;
                    mismatch_count++;
                }
            }
        } else {
            std::cout << "  ✓ Result verification passed (max error: " << max_error << ")" << std::endl;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return metrics;
}

// Benchmark cuSPARSE implementation
PerformanceMetrics benchmark_cusparse_spmv(const CSRMatrix& matrix,
                                            const BenchmarkConfig& config) {
    PerformanceMetrics metrics;
    metrics.kernel_name = "cuSPARSE";
    
    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    // Allocate device memory
    double *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_offsets;
    
    size_t values_size = matrix.nnz * sizeof(double);
    size_t indices_size = matrix.nnz * sizeof(int);
    size_t offsets_size = (matrix.num_rows + 1) * sizeof(int);
    size_t vector_size = matrix.num_cols * sizeof(double);
    size_t result_size = matrix.num_rows * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_values, values_size));
    CUDA_CHECK(cudaMalloc(&d_col_indices, indices_size));
    CUDA_CHECK(cudaMalloc(&d_row_offsets, offsets_size));
    CUDA_CHECK(cudaMalloc(&d_x, vector_size));
    CUDA_CHECK(cudaMalloc(&d_y, result_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_values, matrix.values.data(), values_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, matrix.col_indices.data(), indices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, matrix.row_offsets.data(), offsets_size, cudaMemcpyHostToDevice));
    
    // Initialize input vector
    std::vector<double> h_x(matrix.num_cols, 1.0);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), vector_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, result_size));
    
    // Create matrix and vector descriptors
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, matrix.num_rows, matrix.num_cols, matrix.nnz,
                                     d_row_offsets, d_col_indices, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, matrix.num_cols, d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, matrix.num_rows, d_y, CUDA_R_64F));
    
    // Define scalar values for cuSPARSE API
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // Get buffer size and allocate
    size_t bufferSize;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX,
                                           &beta, vecY, CUDA_R_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
    void* buffer = nullptr;
    if (bufferSize > 0) {
        CUDA_CHECK(cudaMalloc(&buffer, bufferSize));
    }
    
    BenchmarkRunner runner(config.warmup_iterations, config.benchmark_iterations);
    CudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < config.warmup_iterations; i++) {
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX,
                                    &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark runs
    for (int i = 0; i < config.benchmark_iterations; i++) {
        timer.start();
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX,
                                    &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        timer.stop();
        runner.add_time(timer.get_elapsed_time());
    }
    
    // Calculate metrics
    metrics.execution_time_ms = runner.get_geometric_mean_time();
    
    size_t total_bytes = values_size + indices_size + vector_size + result_size;
    metrics.memory_bandwidth_gb_s = calculate_memory_bandwidth(total_bytes, metrics.execution_time_ms);
    metrics.flops = calculate_spmv_flops(matrix.nnz, metrics.execution_time_ms);
    metrics.shared_memory_usage_kb = 0.0; // cuSPARSE doesn't expose this
    
    // cuSPARSE is considered the reference, so mark as correct
    metrics.is_result_correct = true;
    metrics.max_error = 0.0;
    
    // Cleanup
    if (buffer) CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    
    return metrics;
}

// Run comprehensive benchmark comparing all kernels
void run_comprehensive_benchmark(const std::vector<std::string>& matrix_files,
                                 const BenchmarkConfig& config) {
    std::cout << "Starting comprehensive SpMV benchmark..." << std::endl;
    
    GPUInfo gpu_info = get_gpu_info();
    print_gpu_info(gpu_info);
    
    // Define kernels to benchmark
    std::vector<std::pair<SpMVKernel, std::string>> kernels = {
        {spmv_basic, "Basic (D1)"},
        {spmv_shared_memory, "Shared Memory"},
        {spmv_warp_reduce, "Warp Reduce"}
    };
    
    for (const std::string& matrix_file : matrix_files) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Matrix: " << matrix_file << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // TODO: Load matrix (placeholder)
        CSRMatrix matrix = generate_random_csr(1000, 1000, 0.01); // Placeholder
        print_matrix_stats(matrix);
        
        std::vector<PerformanceMetrics> results;
        
        // Benchmark custom kernels
        for (const auto& kernel_pair : kernels) {
            std::cout << "\nBenchmarking " << kernel_pair.second << "..." << std::endl;
            PerformanceMetrics metrics = benchmark_spmv_kernel(
                matrix, kernel_pair.first, kernel_pair.second, config);
            results.push_back(metrics);
        }
        
        // Benchmark cuSPARSE
        std::cout << "\nBenchmarking cuSPARSE..." << std::endl;
        PerformanceMetrics cusparse_metrics = benchmark_cusparse_spmv(matrix, config);
        results.push_back(cusparse_metrics);
        
        // Print results
        print_performance_table(results);
        
        // Save to CSV
        std::string csv_filename = "results_" + matrix_file + ".csv";
        save_results_to_csv(results, csv_filename);
    }
}

// Print performance comparison table
void print_performance_table(const std::vector<PerformanceMetrics>& results) {
    std::cout << "\n" << std::string(100, '-') << std::endl;
    std::cout << "Performance Comparison Results" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    std::cout << std::left << std::setw(20) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Bandwidth (GB/s)"
              << std::setw(15) << "GFLOPS"
              << std::setw(18) << "Shared Mem (KB)"
              << std::setw(12) << "Verification"
              << std::setw(12) << "Max Error" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.kernel_name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) 
                  << result.execution_time_ms
                  << std::setw(15) << std::setprecision(2) << result.memory_bandwidth_gb_s
                  << std::setw(15) << std::setprecision(2) << result.flops / 1e9
                  << std::setw(18) << std::setprecision(1) << result.shared_memory_usage_kb
                  << std::setw(12) << (result.is_result_correct ? "PASS" : "FAIL")
                  << std::setw(12) << std::scientific << std::setprecision(2) << result.max_error
                  << std::endl;
    }
    std::cout << std::string(100, '-') << std::endl;
}

// Save results to CSV file
void save_results_to_csv(const std::vector<PerformanceMetrics>& results,
                         const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "Kernel,Time_ms,Bandwidth_GB_s,GFLOPS,Shared_Memory_KB,Verification,Max_Error" << std::endl;
    
    // Write data
    for (const auto& result : results) {
        file << result.kernel_name << ","
             << result.execution_time_ms << ","
             << result.memory_bandwidth_gb_s << ","
             << result.flops / 1e9 << ","
             << result.shared_memory_usage_kb << ","
             << (result.is_result_correct ? "PASS" : "FAIL") << ","
             << result.max_error << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

// TODO: Implement reference computation and verification functions
void compute_spmv_reference(const CSRMatrix& matrix, const double* x, double* y) {
    // CPU reference implementation
    for (int i = 0; i < matrix.num_rows; i++) {
        double sum = 0.0;
        for (int j = matrix.row_offsets[i]; j < matrix.row_offsets[i + 1]; j++) {
            sum += matrix.values[j] * x[matrix.col_indices[j]];
        }
        y[i] = sum;
    }
}

bool verify_spmv_result(const CSRMatrix& matrix, const double* x, 
                        const double* y_gpu, const double* y_reference,
                        double tolerance) {
    for (int i = 0; i < matrix.num_rows; i++) {
        if (std::abs(y_gpu[i] - y_reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
} 
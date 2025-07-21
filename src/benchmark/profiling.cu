#include "benchmark/benchmark.h"
#include "utils/cuda_utils.h"
#include <iostream>

// Enable profiling for nvprof/nsight compute
void enable_profiling() {
    // This function can be used to insert profiling markers
    // when using nvprof or nsight compute
    
    std::cout << "Profiling mode enabled. Use nvprof or nsight compute for detailed analysis." << std::endl;
    std::cout << "Example commands:" << std::endl;
    std::cout << "  nvprof --metrics all ./benchmark" << std::endl;
    std::cout << "  ncu --metrics all ./benchmark" << std::endl;
}

// Profile a specific kernel with detailed metrics
void profile_kernel_detailed(const CSRMatrix& matrix, SpMVKernel kernel, 
                             const std::string& kernel_name) {
    std::cout << "Profiling kernel: " << kernel_name << std::endl;
    
    // TODO: Add NVTX markers for better profiling
    // nvtxRangePush(kernel_name.c_str());
    
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
    
    // Run kernel for profiling
    std::cout << "Running kernel for profiling..." << std::endl;
    kernel(matrix.num_rows, d_values, d_col_indices, d_row_offsets, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    // nvtxRangePop();
    std::cout << "Profiling completed for " << kernel_name << std::endl;
}

// Print compilation information
void print_compilation_info() {
    std::cout << "\nCompilation Information:" << std::endl;
    std::cout << "  CUDA Compute Capability: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
    
#ifdef __OPTIMIZE__
    std::cout << "  Optimization: Enabled" << std::endl;
#else
    std::cout << "  Optimization: Disabled" << std::endl;
#endif

#ifdef NDEBUG
    std::cout << "  Debug Mode: Disabled" << std::endl;
#else
    std::cout << "  Debug Mode: Enabled" << std::endl;
#endif

    std::cout << "  Recommended compilation flags:" << std::endl;
    std::cout << "    -O3 -use_fast_math -lineinfo" << std::endl;
    std::cout << "    -gencode arch=compute_XX,code=sm_XX (replace XX with your GPU arch)" << std::endl;
} 
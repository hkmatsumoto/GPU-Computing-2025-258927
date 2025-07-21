#include "utils/matrix_io.h"
#include "utils/cuda_utils.h"
#include "utils/timer.h"
#include "kernels/spmv_kernels.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_random_matrix_generation() {
    std::cout << "Testing random matrix generation..." << std::endl;
    
    CSRMatrix matrix = generate_random_csr(100, 100, 0.1);
    
    assert(matrix.num_rows == 100);
    assert(matrix.num_cols == 100);
    assert(matrix.nnz > 0);
    assert(validate_csr_format(matrix));
    
    std::cout << "✓ Random matrix generation test passed" << std::endl;
}

void test_basic_spmv_kernel() {
    std::cout << "Testing basic SpMV kernel..." << std::endl;
    
    // Create small test matrix
    CSRMatrix matrix = generate_random_csr(10, 10, 0.5);
    
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
    
    // Run kernel
    spmv_basic(matrix.num_rows, d_values, d_col_indices, d_row_offsets, d_x, d_y);
    
    // Copy result back
    std::vector<double> h_y(matrix.num_rows);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, result_size, cudaMemcpyDeviceToHost));
    
    // Compute reference on CPU
    std::vector<double> h_y_ref(matrix.num_rows, 0.0);
    for (int i = 0; i < matrix.num_rows; i++) {
        for (int j = matrix.row_offsets[i]; j < matrix.row_offsets[i + 1]; j++) {
            h_y_ref[i] += matrix.values[j] * h_x[matrix.col_indices[j]];
        }
    }
    
    // Verify results
    const double tolerance = 1e-12;
    for (int i = 0; i < matrix.num_rows; i++) {
        if (std::abs(h_y[i] - h_y_ref[i]) > tolerance) {
            std::cerr << "Mismatch at row " << i << ": GPU=" << h_y[i] 
                      << ", CPU=" << h_y_ref[i] << std::endl;
            assert(false);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    std::cout << "✓ Basic SpMV kernel test passed" << std::endl;
}

void test_cuda_utilities() {
    std::cout << "Testing CUDA utilities..." << std::endl;
    
    // Test GPU info
    GPUInfo info = get_gpu_info();
    assert(!info.name.empty());
    assert(info.total_memory > 0);
    assert(info.multiprocessor_count > 0);
    assert(info.warp_size > 0);
    
    // Test launch config calculation
    int grid_size, block_size;
    calculate_launch_config(1000, grid_size, block_size);
    assert(grid_size > 0);
    assert(block_size > 0);
    
    // Test bandwidth calculation
    double bandwidth = calculate_memory_bandwidth(1024*1024*1024, 100.0); // 1GB in 100ms
    assert(bandwidth > 0);
    
    // Test FLOPS calculation
    double flops = calculate_spmv_flops(1000000, 10.0); // 1M nnz in 10ms
    assert(flops > 0);
    
    std::cout << "✓ CUDA utilities test passed" << std::endl;
}

void test_timer_functionality() {
    std::cout << "Testing timer functionality..." << std::endl;
    
    // Test CUDA timer
    CudaTimer cuda_timer;
    cuda_timer.start();
    
    // Simple kernel launch for timing
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1000 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, 1000 * sizeof(int)));
    
    cuda_timer.stop();
    float elapsed = cuda_timer.get_elapsed_time();
    assert(elapsed >= 0);
    
    CUDA_CHECK(cudaFree(d_data));
    
    // Test CPU timer
    CPUTimer cpu_timer;
    cpu_timer.start();
    
    // Small delay
    for (volatile int i = 0; i < 1000000; i++) {}
    
    cpu_timer.stop();
    double cpu_elapsed = cpu_timer.get_elapsed_time();
    assert(cpu_elapsed >= 0);
    
    std::cout << "✓ Timer functionality test passed" << std::endl;
}

int main() {
    std::cout << "Running Deliverable 2 Tests" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    try {
        test_random_matrix_generation();
        test_cuda_utilities();
        test_timer_functionality();
        test_basic_spmv_kernel();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        std::cout << "\nProject setup is ready for Deliverable 2 implementation." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
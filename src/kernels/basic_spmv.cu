#include "kernels/spmv_kernels.h"
#include "utils/cuda_utils.h"

// Basic SpMV kernel from Deliverable 1
__global__ void spmv_basic_kernel(int num_rows, const double* values, 
                                  const int* col_indices, const int* row_offsets,
                                  const double* x, double* y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < num_rows) {
        double sum = 0.0;
        for (int j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}

// Host wrapper function for basic SpMV
void spmv_basic(int num_rows, const double* values, const int* col_indices,
                const int* row_offsets, const double* x, double* y) {
    
    // Calculate launch configuration
    int grid_size, block_size;
    calculate_launch_config(num_rows, grid_size, block_size);
    
    // Launch kernel
    spmv_basic_kernel<<<grid_size, block_size>>>(
        num_rows, values, col_indices, row_offsets, x, y);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} 
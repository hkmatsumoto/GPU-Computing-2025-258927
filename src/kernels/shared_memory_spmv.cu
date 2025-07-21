#include "kernels/spmv_kernels.h"
#include "utils/cuda_utils.h"

// Shared memory optimized SpMV kernel
__global__ void spmv_shared_memory_kernel(int num_rows, const double* values,
                                          const int* col_indices, const int* row_offsets,
                                          const double* x, double* y) {
    // Shared memory for caching x vector elements
    extern __shared__ double shared_x[];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Cache size: number of elements we can fit in shared memory
    int cache_size = blockDim.x; // Conservative: one element per thread
    
    // Simple but effective shared memory caching
    // Cache a sliding window of x vector elements
    int base_idx = blockIdx.x * cache_size;
    
    // Phase 1: Load x elements into shared memory
    if (tid < cache_size && base_idx + tid < num_rows) {
        shared_x[tid] = x[base_idx + tid];
    }
    __syncthreads();
    
    // Phase 2: Compute SpMV with shared memory when possible
    if (row < num_rows) {
        double sum = 0.0;
        for (int j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
            int col = col_indices[j];
            double x_val;
            
            // Use shared memory if the column is in our cached range
            if (col >= base_idx && col < base_idx + cache_size) {
                x_val = shared_x[col - base_idx];
            } else {
                x_val = x[col];
            }
            
            sum += values[j] * x_val;
        }
        y[row] = sum;
    }
}

// Host wrapper function for shared memory SpMV
void spmv_shared_memory(int num_rows, const double* values, const int* col_indices,
                        const int* row_offsets, const double* x, double* y) {
    
    // Calculate launch configuration
    int grid_size, block_size;
    calculate_launch_config(num_rows, grid_size, block_size);
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(double); // TODO: Optimize this
    
    // Launch kernel
    spmv_shared_memory_kernel<<<grid_size, block_size, shared_mem_size>>>(
        num_rows, values, col_indices, row_offsets, x, y);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} 
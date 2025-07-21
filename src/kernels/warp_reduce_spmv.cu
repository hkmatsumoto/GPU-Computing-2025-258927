#include "kernels/spmv_kernels.h"
#include "utils/cuda_utils.h"

#define WARP_SIZE 32

// Warp Reduce SpMV kernel using __shfl_xor_sync
__global__ void spmv_warp_reduce_kernel(int num_rows, const double* values, 
                                        const int* col_indices, const int* row_offsets,
                                        const double* x, double* y) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = global_thread_id / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (row_idx >= num_rows) {
        return;
    }
    
    double sum = 0.0;
    int row_start = row_offsets[row_idx];
    int row_end = row_offsets[row_idx + 1];
    
    // Warp-collaborative processing: each thread processes different elements
    for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
        sum += values[j] * x[col_indices[j]];
    }
    
    // Warp-level reduction using XOR-based shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Only lane 0 writes the final result
    if (lane_id == 0) {
        y[row_idx] = sum;
    }
}

// Host wrapper function for warp reduce SpMV
void spmv_warp_reduce(int num_rows, const double* values, const int* col_indices,
                      const int* row_offsets, const double* x, double* y) {
    
    // Calculate launch configuration
    int grid_size, block_size;
    calculate_launch_config(num_rows, grid_size, block_size);
    
    // Ensure we have enough threads for all rows (each warp processes one row)
    int total_threads_needed = num_rows * WARP_SIZE;
    int total_threads = grid_size * block_size;
    
    if (total_threads < total_threads_needed) {
        grid_size = (total_threads_needed + block_size - 1) / block_size;
    }
    
    // Launch kernel
    spmv_warp_reduce_kernel<<<grid_size, block_size>>>(
        num_rows, values, col_indices, row_offsets, x, y);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} 
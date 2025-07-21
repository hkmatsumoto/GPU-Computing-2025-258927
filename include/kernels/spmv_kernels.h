#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include <cuda_runtime.h>

// Basic SpMV kernel from Deliverable 1
__global__ void spmv_basic_kernel(int num_rows, const double* values, 
                                  const int* col_indices, const int* row_offsets,
                                  const double* x, double* y);

// Shared memory optimized SpMV kernel
__global__ void spmv_shared_memory_kernel(int num_rows, const double* values,
                                          const int* col_indices, const int* row_offsets,
                                          const double* x, double* y);

// Warp Reduce kernel using XOR-based shuffle reduction
__global__ void spmv_warp_reduce_kernel(int num_rows, const double* values,
                                        const int* col_indices, const int* row_offsets,
                                        const double* x, double* y);

// Host wrapper functions for easier calling
void spmv_basic(int num_rows, const double* values, const int* col_indices,
                const int* row_offsets, const double* x, double* y);

void spmv_shared_memory(int num_rows, const double* values, const int* col_indices,
                        const int* row_offsets, const double* x, double* y);

void spmv_warp_reduce(int num_rows, const double* values, const int* col_indices,
                      const int* row_offsets, const double* x, double* y);

#endif // SPMV_KERNELS_H 
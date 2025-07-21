#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

// cuSPARSE error checking macro
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << status << std::endl; \
            exit(1); \
        } \
    } while (0)

// GPU device info
struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int warp_size;
};

// Get GPU information
GPUInfo get_gpu_info();

// Print GPU information
void print_gpu_info(const GPUInfo& info);

// Calculate optimal grid and block dimensions
void calculate_launch_config(int num_elements, int& grid_size, int& block_size);

// Memory bandwidth calculation
double calculate_memory_bandwidth(size_t bytes_transferred, double time_ms);

// FLOPS calculation for SpMV
double calculate_spmv_flops(int nnz, double time_ms);

#endif // CUDA_UTILS_H 
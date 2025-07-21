#include "utils/cuda_utils.h"
#include <iomanip>

// Get GPU information
GPUInfo get_gpu_info() {
    GPUInfo info;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDevice(&info.device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, info.device_id));
    
    info.name = std::string(prop.name);
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_shared_memory_per_block = prop.sharedMemPerBlock;
    info.warp_size = prop.warpSize;
    
    return info;
}

// Print GPU information
void print_gpu_info(const GPUInfo& info) {
    std::cout << "GPU Information:" << std::endl;
    std::cout << "  Device ID: " << info.device_id << std::endl;
    std::cout << "  Name: " << info.name << std::endl;
    std::cout << "  Total Memory: " << std::fixed << std::setprecision(2) 
              << info.total_memory / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "  Multiprocessors: " << info.multiprocessor_count << std::endl;
    std::cout << "  Max Threads per Block: " << info.max_threads_per_block << std::endl;
    std::cout << "  Max Shared Memory per Block: " 
              << info.max_shared_memory_per_block / 1024 << " KB" << std::endl;
    std::cout << "  Warp Size: " << info.warp_size << std::endl;
}

// Calculate optimal grid and block dimensions
void calculate_launch_config(int num_elements, int& grid_size, int& block_size) {
    // Use a common block size that works well for most kernels
    block_size = 256;
    
    // Calculate grid size to cover all elements
    grid_size = (num_elements + block_size - 1) / block_size;
    
    // Limit grid size to avoid excessive overhead
    const int max_grid_size = 65535;
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }
}

// Calculate memory bandwidth in GB/s
double calculate_memory_bandwidth(size_t bytes_transferred, double time_ms) {
    // Convert to GB/s: bytes -> GB, ms -> s
    double gb_transferred = bytes_transferred / (1024.0 * 1024.0 * 1024.0);
    double time_s = time_ms / 1000.0;
    
    return gb_transferred / time_s;
}

// Calculate FLOPS for SpMV (2 * nnz operations: multiply + add)
double calculate_spmv_flops(int nnz, double time_ms) {
    double operations = 2.0 * nnz; // multiply + add for each non-zero
    double time_s = time_ms / 1000.0;
    
    return operations / time_s; // FLOPS
} 
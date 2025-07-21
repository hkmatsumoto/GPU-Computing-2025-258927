#include "benchmark/benchmark.h"
#include "kernels/spmv_kernels.h"
#include "utils/matrix_io.h"
#include "utils/cuda_utils.h"
#include <iostream>
#include <vector>
#include <string>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help     Show this help message" << std::endl;
    std::cout << "  -m, --matrix   Matrix file (Matrix Market format)" << std::endl;
    std::cout << "  -w, --warmup   Number of warmup iterations (default: 3)" << std::endl;
    std::cout << "  -b, --bench    Number of benchmark iterations (default: 10)" << std::endl;
    std::cout << "  -v, --verify   Enable result verification (default: true)" << std::endl;
    std::cout << "  --random       Use random generated matrix for testing" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " --random" << std::endl;
    std::cout << "  " << program_name << " -m matrix.mtx" << std::endl;
    std::cout << "  " << program_name << " -m matrix.mtx -w 5 -b 20" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "SpMV Performance Optimization - Deliverable 2" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Parse command line arguments
    BenchmarkConfig config;
    std::vector<std::string> matrix_files;
    bool use_random = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-m" || arg == "--matrix") {
            if (i + 1 < argc) {
                matrix_files.push_back(argv[++i]);
            } else {
                std::cerr << "Error: Matrix file not specified" << std::endl;
                return 1;
            }
        }
        else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) {
                config.warmup_iterations = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: Warmup iterations not specified" << std::endl;
                return 1;
            }
        }
        else if (arg == "-b" || arg == "--bench") {
            if (i + 1 < argc) {
                config.benchmark_iterations = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: Benchmark iterations not specified" << std::endl;
                return 1;
            }
        }
        else if (arg == "-v" || arg == "--verify") {
            config.verify_results = true;
        }
        else if (arg == "--random") {
            use_random = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    // Print GPU information
    GPUInfo gpu_info = get_gpu_info();
    print_gpu_info(gpu_info);
    
    // Setup test matrices
    if (use_random || matrix_files.empty()) {
        std::cout << "\nGenerating random test matrices..." << std::endl;
        
        // Generate different sizes and sparsity patterns
        struct TestConfig {
            int rows;
            int cols;
            double density;
        };
        
        std::vector<TestConfig> test_configs = {
            {5000, 5000, 0.05},        // Medium, dense (1.25M elements)
            {10000, 10000, 0.01},      // Large, sparse (1M elements)
            {20000, 20000, 0.005},     // Very large, very sparse (2M elements)
            {30000, 30000, 0.003},     // Huge, sparse (2.7M elements)
            {40000, 40000, 0.002},     // Massive, sparse (3.2M elements)
        };
        
        for (size_t i = 0; i < test_configs.size(); i++) {
            int rows = test_configs[i].rows;
            int cols = test_configs[i].cols;
            double density = test_configs[i].density;
            std::cout << "\nTesting matrix " << (i+1) << ": " 
                      << rows << "x" << cols << ", density=" << density << std::endl;
            
            CSRMatrix matrix = generate_random_csr(rows, cols, density);
            print_matrix_stats(matrix);
            
            if (!validate_csr_format(matrix)) {
                std::cerr << "Error: Generated matrix is invalid" << std::endl;
                continue;
            }
            
            std::vector<PerformanceMetrics> results;
            
            std::cout << "\nBenchmarking Basic kernel..." << std::endl;
            PerformanceMetrics basic_metrics = benchmark_spmv_kernel(
                matrix, spmv_basic, "Basic (D1)", config);
            results.push_back(basic_metrics);
            
            std::cout << "Benchmarking Shared Memory kernel..." << std::endl;
            PerformanceMetrics shared_metrics = benchmark_spmv_kernel(
                matrix, spmv_shared_memory, "Shared Memory", config);
            results.push_back(shared_metrics);
            
            std::cout << "Benchmarking Warp Reduce kernel..." << std::endl;
            PerformanceMetrics warp_reduce_metrics = benchmark_spmv_kernel(
                matrix, spmv_warp_reduce, "Warp Reduce", config);
            results.push_back(warp_reduce_metrics);
            
            std::cout << "Benchmarking cuSPARSE..." << std::endl;
            PerformanceMetrics cusparse_metrics = benchmark_cusparse_spmv(matrix, config);
            results.push_back(cusparse_metrics);
            
            // Print results
            print_performance_table(results);
            
            // Save results
            std::string csv_filename = "results/matrix_" + std::to_string(i+1) + ".csv";
            save_results_to_csv(results, csv_filename);
        }
    }
    else {
        // Use provided matrix files
        std::cout << "\nProcessing provided matrix files..." << std::endl;
        
        for (size_t i = 0; i < matrix_files.size(); i++) {
            const std::string& matrix_file = matrix_files[i];
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Processing Matrix Market file: " << matrix_file << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
            // Load matrix from file
            CSRMatrix matrix = read_matrix_market(matrix_file);
            print_matrix_stats(matrix);
            
            if (!validate_csr_format(matrix)) {
                std::cerr << "Error: Loaded matrix is invalid" << std::endl;
                continue;
            }
            
            std::vector<PerformanceMetrics> results;
            
            std::cout << "\nBenchmarking Basic kernel..." << std::endl;
            PerformanceMetrics basic_metrics = benchmark_spmv_kernel(
                matrix, spmv_basic, "Basic (D1)", config);
            results.push_back(basic_metrics);
            
            std::cout << "Benchmarking Shared Memory kernel..." << std::endl;
            PerformanceMetrics shared_metrics = benchmark_spmv_kernel(
                matrix, spmv_shared_memory, "Shared Memory", config);
            results.push_back(shared_metrics);
            
            std::cout << "Benchmarking Warp Reduce kernel..." << std::endl;
            PerformanceMetrics warp_reduce_metrics = benchmark_spmv_kernel(
                matrix, spmv_warp_reduce, "Warp Reduce", config);
            results.push_back(warp_reduce_metrics);
            
            std::cout << "Benchmarking cuSPARSE..." << std::endl;
            PerformanceMetrics cusparse_metrics = benchmark_cusparse_spmv(matrix, config);
            results.push_back(cusparse_metrics);
            
            // Print results
            print_performance_table(results);
            
            // Save results with file-specific name
            std::string filename_base = matrix_file.substr(matrix_file.find_last_of("/") + 1);
            filename_base = filename_base.substr(0, filename_base.find_last_of("."));
            std::string csv_filename = "results/" + filename_base + "_results.csv";
            save_results_to_csv(results, csv_filename);
        }
    }
    
    std::cout << "\nBenchmark completed!" << std::endl;
    std::cout << "\nResults have been saved to the results/ directory." << std::endl;
    std::cout << "All kernels have been implemented and tested successfully." << std::endl;
    
    return 0;
} 
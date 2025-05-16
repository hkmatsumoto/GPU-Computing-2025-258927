#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Structure for CSR (Compressed Sparse Row) format
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    double* values;      // Non-zero values
    int* col_indices;    // Column indices
    int* row_offsets;    // Row offsets
    bool is_gpu;         // Flag to track if memory is on GPU

    CSRMatrix() : values(nullptr), col_indices(nullptr), row_offsets(nullptr), is_gpu(false) {}

    // Destructor to handle memory cleanup
    ~CSRMatrix() {
        if (values) {
            if (is_gpu) {
                cudaFree(values);
                cudaFree(col_indices);
                cudaFree(row_offsets);
            } else {
                delete[] values;
                delete[] col_indices;
                delete[] row_offsets;
            }
        }
    }
};

// Error checking macro
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Function to generate random vector
double* generate_random_vector(int size) {
    double* vec = new double[size];
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        vec[i] = distribution(generator);
    }
    
    return vec;
}

// Function to perform SpMV on CPU (y = Ax)
void spmv_cpu(const CSRMatrix& A, const double* x, double* y) {
    for (int i = 0; i < A.num_rows; i++) {
        y[i] = 0.0;
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

// CUDA kernel for SpMV
__global__ void spmv_kernel(int num_rows, const double* values, const int* col_indices, 
                           const int* row_offsets, const double* x, double* y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < num_rows) {
        double sum = 0.0;
        for (int j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}

// Function to perform SpMV on GPU
void spmv_gpu(const CSRMatrix& A, const double* x_gpu, double* y_gpu) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (A.num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    spmv_kernel<<<num_blocks, BLOCK_SIZE>>>(A.num_rows, A.values, 
        A.col_indices, A.row_offsets, x_gpu, y_gpu);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Function to transfer matrix to GPU
CSRMatrix transfer_matrix_to_gpu(const CSRMatrix& A) {
    CSRMatrix A_gpu;
    A_gpu.num_rows = A.num_rows;
    A_gpu.num_cols = A.num_cols;
    A_gpu.num_nonzeros = A.num_nonzeros;
    A_gpu.is_gpu = true;
    
    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&A_gpu.values, A.num_nonzeros * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&A_gpu.col_indices, A.num_nonzeros * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_gpu.row_offsets, (A.num_rows + 1) * sizeof(int)));
    
    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(A_gpu.values, A.values, 
        A.num_nonzeros * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_gpu.col_indices, A.col_indices, 
        A.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_gpu.row_offsets, A.row_offsets, 
        (A.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    return A_gpu;
}

// Function to read MTX file and convert it to CSR format
CSRMatrix read_mtx_to_csr(const char* filename) {
    std::ifstream file(filename);
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    CSRMatrix matrix;
    std::string line;

    // Skip header lines (starting with %%)
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    // Read dimensions and number of non-zero elements
    std::istringstream iss(line);
    iss >> matrix.num_rows >> matrix.num_cols >> matrix.num_nonzeros;

    // Verify matrix dimensions
    if (matrix.num_rows <= 0 || matrix.num_cols <= 0 || matrix.num_nonzeros <= 0) {
        printf("Invalid matrix dimensions or non-zero count\n");
        exit(1);
    }

    // Temporary vector for COO format
    std::vector<std::tuple<int, int, double>> coo_data;
    coo_data.reserve(matrix.num_nonzeros);

    // Read elements from file
    int row, col;
    double value;
    int count = 0;
    while (file >> row >> col >> value) {
        // Verify indices
        if (row < 1 || row > matrix.num_rows || col < 1 || col > matrix.num_cols) {
            printf("Invalid index at element %d: row=%d, col=%d\n", count + 1, row, col);
            exit(1);
        }
        // Convert from 1-based to 0-based indexing
        coo_data.push_back(std::make_tuple(row-1, col-1, value));
        count++;
    }

    // Allocate arrays
    matrix.values = new double[matrix.num_nonzeros];
    matrix.col_indices = new int[matrix.num_nonzeros];
    matrix.row_offsets = new int[matrix.num_rows + 1]();  // Initialize to zero
    matrix.is_gpu = false;

    // Count number of elements in each row
    for (const auto& elem : coo_data) {
        matrix.row_offsets[std::get<0>(elem) + 1]++;
    }

    // Calculate cumulative sum for row_offsets
    for (int i = 1; i <= matrix.num_rows; i++) {
        matrix.row_offsets[i] += matrix.row_offsets[i-1];
    }

    // Place elements in their correct positions
    std::vector<int> row_counts(matrix.num_rows, 0);
    for (const auto& elem : coo_data) {
        int row = std::get<0>(elem);
        int pos = matrix.row_offsets[row] + row_counts[row];
        
        matrix.values[pos] = std::get<2>(elem);
        matrix.col_indices[pos] = std::get<1>(elem);
        row_counts[row]++;
    }

    return matrix;
}

// Function to verify results
bool verify_results(const double* cpu_result, const double* gpu_result, int size,
                   double tolerance = 1e-6) {
    for (int i = 0; i < size; i++) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Read MTX file and convert to CSR format
    CSRMatrix matrix = read_mtx_to_csr(argv[1]);
    printf("Matrix loaded: %d rows, %d columns, %d non-zeros\n", 
           matrix.num_rows, matrix.num_cols, matrix.num_nonzeros);

    // Generate input vector and allocate output vectors
    double* x = generate_random_vector(matrix.num_cols);
    double* y_cpu = new double[matrix.num_rows];
    double* y_gpu = new double[matrix.num_rows];

    // CPU SpMV
    auto cpu_start = std::chrono::high_resolution_clock::now();
    spmv_cpu(matrix, x, y_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    // Transfer matrix to GPU
    CSRMatrix matrix_gpu = transfer_matrix_to_gpu(matrix);

    // Allocate and copy vectors on GPU
    double *x_gpu, *y_gpu_dev;
    CHECK_CUDA(cudaMalloc(&x_gpu, matrix.num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_gpu_dev, matrix.num_rows * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(x_gpu, x, matrix.num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // GPU SpMV
    auto gpu_start = std::chrono::high_resolution_clock::now();
    spmv_gpu(matrix_gpu, x_gpu, y_gpu_dev);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(y_gpu, y_gpu_dev, matrix.num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify results
    bool results_match = verify_results(y_cpu, y_gpu, matrix.num_rows);

    // Calculate GFLOPS
    double cpu_gflops = (2.0 * matrix.num_nonzeros) / (cpu_duration.count() * 1000.0);
    double gpu_gflops = (2.0 * matrix.num_nonzeros) / (gpu_duration.count() * 1000.0);

    // Print human-readable output
    printf("\nPerformance Results:\n");
    printf("------------------\n");
    printf("CPU time: %ld us (%.2f GFLOPS)\n", cpu_duration.count(), cpu_gflops);
    printf("GPU time: %ld us (%.2f GFLOPS)\n", gpu_duration.count(), gpu_gflops);
    printf("Validation: Results %s\n\n", results_match ? "match" : "do not match");

    // Output machine-readable JSON format
    printf("JSON_START\n");  // Marker for parsing
    printf("{\n");
    printf("  \"matrix\": {\n");
    printf("    \"rows\": %d,\n", matrix.num_rows);
    printf("    \"cols\": %d,\n", matrix.num_cols);
    printf("    \"nnz\": %d\n", matrix.num_nonzeros);
    printf("  },\n");
    printf("  \"cpu\": {\n");
    printf("    \"time_us\": %ld,\n", cpu_duration.count());
    printf("    \"gflops\": %.2f\n", cpu_gflops);
    printf("  },\n");
    printf("  \"gpu\": {\n");
    printf("    \"time_us\": %ld,\n", gpu_duration.count());
    printf("    \"gflops\": %.2f\n", gpu_gflops);
    printf("  },\n");
    printf("  \"validation\": {\n");
    printf("    \"results_match\": %s\n", results_match ? "true" : "false");
    printf("  }\n");
    printf("}\n");
    printf("JSON_END\n");  // Marker for parsing

    // Cleanup
    delete[] x;
    delete[] y_cpu;
    delete[] y_gpu;

    return 0;
}
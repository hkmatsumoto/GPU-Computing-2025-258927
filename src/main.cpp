#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <random>
#include <chrono>

// Structure for CSR (Compressed Sparse Row) format
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    std::vector<double> values;      // Non-zero values
    std::vector<int> col_indices;    // Column indices
    std::vector<int> row_offsets;    // Row offsets
};

// Function to generate random vector
std::vector<double> generate_random_vector(int size) {
    std::vector<double> vec(size);
    
    // Use current time as seed for random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        vec[i] = distribution(generator);
    }
    
    return vec;
}

// Function to perform SpMV on CPU (y = Ax)
void spmv_cpu(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < A.num_rows; i++) {
        y[i] = 0.0;
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

// Function to print vector (for debugging)
void print_vector(const std::vector<double>& vec, const char* name) {
    printf("%s: ", name);
    for (size_t i = 0; i < vec.size(); i++) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
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

    // Temporary vector for COO format
    std::vector<std::tuple<int, int, double>> coo_data;
    coo_data.reserve(matrix.num_nonzeros);

    // Read elements from file
    int row, col;
    double value;
    while (file >> row >> col >> value) {
        // Convert from 1-based to 0-based indexing
        coo_data.push_back(std::make_tuple(row-1, col-1, value));
    }

    // Convert from COO to CSR format
    matrix.values.resize(matrix.num_nonzeros);
    matrix.col_indices.resize(matrix.num_nonzeros);
    matrix.row_offsets.resize(matrix.num_rows + 1, 0);

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

// Debug function to print CSR matrix
void print_csr_matrix(const CSRMatrix& matrix) {
    printf("Matrix dimensions: %d x %d with %d non-zero elements\n",
           matrix.num_rows, matrix.num_cols, matrix.num_nonzeros);
    
    printf("\nValues: ");
    for (int i = 0; i < matrix.num_nonzeros; i++) {
        printf("%.2f ", matrix.values[i]);
    }
    
    printf("\nColumn indices: ");
    for (int i = 0; i < matrix.num_nonzeros; i++) {
        printf("%d ", matrix.col_indices[i]);
    }
    
    printf("\nRow offsets: ");
    for (int i = 0; i <= matrix.num_rows; i++) {
        printf("%d ", matrix.row_offsets[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Read MTX file and convert to CSR format
    CSRMatrix matrix = read_mtx_to_csr(argv[1]);
    
    // Print matrix information for debugging
    print_csr_matrix(matrix);

    // Generate random input vector
    std::vector<double> x = generate_random_vector(matrix.num_cols);
    std::vector<double> y(matrix.num_rows);

    // Print input vector
    print_vector(x, "Input vector x");

    // Perform SpMV on CPU
    spmv_cpu(matrix, x, y);

    // Print result vector
    print_vector(y, "Result vector y");

    return 0;
} 
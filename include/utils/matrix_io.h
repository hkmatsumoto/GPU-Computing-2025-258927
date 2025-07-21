#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <vector>
#include <string>

// CSR format matrix structure
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_offsets;
};

// Function to read matrix from Matrix Market format
CSRMatrix read_matrix_market(const std::string& filename);

// Function to generate random CSR matrix for testing
CSRMatrix generate_random_csr(int num_rows, int num_cols, double density);

// Function to print matrix statistics
void print_matrix_stats(const CSRMatrix& matrix);

// Function to validate CSR format
bool validate_csr_format(const CSRMatrix& matrix);

#endif // MATRIX_IO_H 
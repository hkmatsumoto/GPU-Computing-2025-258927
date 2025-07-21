#include "utils/matrix_io.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include <set>
#include <tuple>

// Read matrix from Matrix Market format
CSRMatrix read_matrix_market(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    CSRMatrix matrix;
    std::string line;
    
    // Skip header comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    // Read matrix dimensions
    std::stringstream ss(line);
    ss >> matrix.num_rows >> matrix.num_cols >> matrix.nnz;
    
    std::cout << "Reading Matrix Market file: " << filename << std::endl;
    std::cout << "Dimensions: " << matrix.num_rows << "x" << matrix.num_cols 
              << ", NNZ: " << matrix.nnz << std::endl;
    
    // Read coordinate format entries (COO)
    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(matrix.nnz);
    
    for (int i = 0; i < matrix.nnz; i++) {
        int row, col;
        double val;
        if (!(file >> row >> col >> val)) {
            std::cerr << "Error reading entry " << i << std::endl;
            exit(1);
        }
        // Convert to 0-based indexing (Matrix Market uses 1-based)
        entries.emplace_back(row - 1, col - 1, val);
    }
    
    file.close();
    
    // Sort entries by row, then by column
    std::sort(entries.begin(), entries.end());
    
    // Convert COO to CSR format
    matrix.values.resize(matrix.nnz);
    matrix.col_indices.resize(matrix.nnz);
    matrix.row_offsets.resize(matrix.num_rows + 1, 0);
    
    // Fill CSR arrays
    int current_row = 0;
    for (int i = 0; i < matrix.nnz; i++) {
        int row = std::get<0>(entries[i]);
        int col = std::get<1>(entries[i]);
        double val = std::get<2>(entries[i]);
        
        // Update row_offsets for new rows
        while (current_row <= row) {
            matrix.row_offsets[current_row] = i;
            current_row++;
        }
        
        matrix.values[i] = val;
        matrix.col_indices[i] = col;
    }
    
    // Set final row offset
    matrix.row_offsets[matrix.num_rows] = matrix.nnz;
    
    std::cout << "Matrix Market file loaded successfully" << std::endl;
    return matrix;
}

// Generate random CSR matrix for testing
CSRMatrix generate_random_csr(int num_rows, int num_cols, double density) {
    CSRMatrix matrix;
    matrix.num_rows = num_rows;
    matrix.num_cols = num_cols;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    matrix.row_offsets.resize(num_rows + 1, 0);
    
    // Generate entries row by row with proper probability distribution
    std::vector<double> temp_values;
    std::vector<int> temp_col_indices;
    
    int current_nnz = 0;
    for (int i = 0; i < num_rows; i++) {
        matrix.row_offsets[i] = current_nnz;
        
        // Use set to avoid duplicate column indices in the same row
        std::set<int> cols_in_row;
        
        // Generate entries for this row based on probability
        for (int j = 0; j < num_cols; j++) {
            if (prob_dist(gen) < density) {
                cols_in_row.insert(j);
            }
        }
        
        // Add entries to temporary vectors (sorted by column index)
        for (int col : cols_in_row) {
            temp_values.push_back(val_dist(gen));
            temp_col_indices.push_back(col);
            current_nnz++;
        }
    }
    
    matrix.row_offsets[num_rows] = current_nnz;
    matrix.nnz = current_nnz;
    
    // Copy to final vectors
    matrix.values = temp_values;
    matrix.col_indices = temp_col_indices;
    
    return matrix;
}

// Print matrix statistics
void print_matrix_stats(const CSRMatrix& matrix) {
    std::cout << "Matrix Statistics:" << std::endl;
    std::cout << "  Rows: " << matrix.num_rows << std::endl;
    std::cout << "  Cols: " << matrix.num_cols << std::endl;
    std::cout << "  NNZ: " << matrix.nnz << std::endl;
    
    double density = static_cast<double>(matrix.nnz) / (matrix.num_rows * matrix.num_cols);
    std::cout << "  Density: " << density * 100 << "%" << std::endl;
    
    // Calculate average entries per row
    double avg_entries = static_cast<double>(matrix.nnz) / matrix.num_rows;
    std::cout << "  Avg entries per row: " << avg_entries << std::endl;
}

// Validate CSR format
bool validate_csr_format(const CSRMatrix& matrix) {
    // Check dimensions
    if (matrix.num_rows <= 0 || matrix.num_cols <= 0 || matrix.nnz < 0) {
        std::cerr << "Invalid matrix dimensions" << std::endl;
        return false;
    }
    
    // Check vector sizes
    if (matrix.values.size() != matrix.nnz ||
        matrix.col_indices.size() != matrix.nnz ||
        matrix.row_offsets.size() != matrix.num_rows + 1) {
        std::cerr << "Inconsistent vector sizes" << std::endl;
        return false;
    }
    
    // Check row offsets
    if (matrix.row_offsets[0] != 0 || matrix.row_offsets[matrix.num_rows] != matrix.nnz) {
        std::cerr << "Invalid row offsets" << std::endl;
        return false;
    }
    
    // Check column indices
    for (int i = 0; i < matrix.nnz; i++) {
        if (matrix.col_indices[i] < 0 || matrix.col_indices[i] >= matrix.num_cols) {
            std::cerr << "Invalid column index: " << matrix.col_indices[i] << std::endl;
            return false;
        }
    }
    
    return true;
} 
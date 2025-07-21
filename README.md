# SpMV Performance Optimization - Deliverable 2

CUDA-based Sparse Matrix-Vector multiplication (SpMV) kernels with performance optimizations.

## Building

```bash
# Load CUDA module (on cluster)
module load CUDA/12.5.0

# Build
make

# Clean
make clean
```

## Usage

### Basic Commands

```bash
# Run with random matrices
./bin/benchmark --random

# Use specific matrix file
./bin/benchmark -m matrix.mtx

# Custom iterations
./bin/benchmark -m matrix.mtx -w 5 -b 20
```

### Options

- `-h, --help`: Show help
- `-m, --matrix`: Matrix file (Matrix Market format)
- `-w, --warmup`: Warmup iterations (default: 3)
- `-b, --bench`: Benchmark iterations (default: 10)
- `-v, --verify`: Enable result verification
- `--random`: Use random matrices

### Profiling

```bash
# Basic profiling
sudo $(which ncu) ./bin/benchmark --random
```

## Kernels

- **Basic**: Baseline implementation
- **Shared Memory**: Shared memory optimization  
- **Warp Reduce**: Warp-collaborative processing
- **cuSPARSE**: Reference implementation

## Results

Results are saved as CSV files in `results/` directory for analysis. 
#!/bin/bash

# Basic SpMV Kernel Profiling Script using NSIGHT COMPUTE
# Usage: ./scripts/profile_basic.sh [matrix_file]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    print_error "NSIGHT COMPUTE (ncu) not found. Please install CUDA toolkit with nsight compute."
    exit 1
fi

# Use sudo for ncu to access GPU Performance Counters
NCU_CMD="sudo $(which ncu)"
print_info "Using: $NCU_CMD (requires sudo for GPU Performance Counters)"

# Default matrix file
MATRIX_FILE=${1:-"data/1138_bus.mtx"}
OUTPUT_DIR="results/profiling"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

print_info "Starting NSIGHT COMPUTE profiling of basic SpMV kernel"
print_info "Matrix file: $MATRIX_FILE"
print_info "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if benchmark exists
if [[ ! -f "bin/benchmark" ]]; then
    print_error "Benchmark executable not found. Please run 'make' first."
    exit 1
fi

print_info "=== Phase 1: Overview Analysis ==="
# Basic overview with common metrics
$NCU_CMD --kernel-name spmv_basic_kernel \
    --launch-skip 3 --launch-count 1 \
    -o "$OUTPUT_DIR/basic_overview_${TIMESTAMP}" \
    --set full \
    ./bin/benchmark -m "$MATRIX_FILE" -w 3 -b 1

print_success "Phase 1 completed. Results saved to basic_overview_${TIMESTAMP}.ncu-rep"

print_info "=== Phase 2: Memory Analysis ==="
# Detailed memory analysis
$NCU_CMD --kernel-name spmv_basic_kernel \
    --launch-skip 3 --launch-count 1 \
    -o "$OUTPUT_DIR/basic_memory_${TIMESTAMP}" \
    --metrics \
dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex_throughput.avg.pct_of_peak_sustained_elapsed,\
l2_throughput.avg.pct_of_peak_sustained_elapsed,\
gld_throughput,gst_throughput,\
gld_efficiency,gst_efficiency,\
gld_transactions,gst_transactions,\
global_load_requests,global_store_requests,\
l1tex_cache_hit_rate,\
l2_cache_hit_rate \
    ./bin/benchmark -m "$MATRIX_FILE" -w 3 -b 1

print_success "Phase 2 completed. Results saved to basic_memory_${TIMESTAMP}.ncu-rep"

print_info "=== Phase 3: Compute Analysis ==="
# Compute efficiency analysis
$NCU_CMD --kernel-name spmv_basic_kernel \
    --launch-skip 3 --launch-count 1 \
    -o "$OUTPUT_DIR/basic_compute_${TIMESTAMP}" \
    --metrics \
sm_efficiency,achieved_occupancy,\
theoretical_occupancy,\
ipc,issued_ipc,\
stall_memory_throttle,stall_memory_dependency,\
stall_inst_fetch,stall_exec_dependency,\
warp_execution_efficiency,\
branch_efficiency \
    ./bin/benchmark -m "$MATRIX_FILE" -w 3 -b 1

print_success "Phase 3 completed. Results saved to basic_compute_${TIMESTAMP}.ncu-rep"

print_info "=== Phase 4: Roofline Analysis ==="
# Roofline model data
$NCU_CMD --kernel-name spmv_basic_kernel \
    --launch-skip 3 --launch-count 1 \
    -o "$OUTPUT_DIR/basic_roofline_${TIMESTAMP}" \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    ./bin/benchmark -m "$MATRIX_FILE" -w 3 -b 1

print_success "Phase 4 completed. Results saved to basic_roofline_${TIMESTAMP}.ncu-rep"

print_info "=== Analysis Summary ==="
echo "Profiling completed! Open the following files in NSIGHT COMPUTE GUI:"
echo "  1. Overview:  $OUTPUT_DIR/basic_overview_${TIMESTAMP}.ncu-rep"
echo "  2. Memory:    $OUTPUT_DIR/basic_memory_${TIMESTAMP}.ncu-rep" 
echo "  3. Compute:   $OUTPUT_DIR/basic_compute_${TIMESTAMP}.ncu-rep"
echo "  4. Roofline:  $OUTPUT_DIR/basic_roofline_${TIMESTAMP}.ncu-rep"
echo ""
echo "Key metrics to examine:"
echo "  • Memory throughput vs peak (should be >80% for memory-bound)"
echo "  • Global load efficiency (should be >90%)"
echo "  • Achieved occupancy (should be >50%)" 
echo "  • IPC (instructions per cycle)"
echo "  • Memory stalls (identify bottleneck)"
echo ""
echo "To view in GUI: nsight-compute [filename].ncu-rep"
echo "To view in terminal: ncu --import [filename].ncu-rep --print-summary"

# Quick terminal summary
print_info "=== Quick Terminal Summary ==="
for file in "$OUTPUT_DIR"/basic_*_${TIMESTAMP}.ncu-rep; do
    if [[ -f "$file" ]]; then
        echo "--- $(basename "$file") ---"
        $NCU_CMD --import "$file" --print-summary 2>/dev/null | head -20
        echo ""
    fi
done 
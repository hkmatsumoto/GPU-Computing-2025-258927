# Deliverable 2 SpMV Optimization Makefile

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA compute capability - update this for your GPU
# Common values: 70 (V100), 75 (RTX 2080), 80 (A100), 86 (RTX 3080), 89 (L40S)
COMPUTE_CAPABILITY = 89

# Compilation flags
NVCC_FLAGS = -O3 -use_fast_math -lineinfo -std=c++11 \
            -gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) \
            -Iinclude

# Libraries
LIBS = -lcusparse -lcublas

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
KERNEL_SOURCES = $(wildcard $(SRC_DIR)/kernels/*.cu)
UTILS_SOURCES = $(wildcard $(SRC_DIR)/utils/*.cu)
BENCHMARK_SOURCES = $(wildcard $(SRC_DIR)/benchmark/*.cu)

ALL_SOURCES = $(KERNEL_SOURCES) $(UTILS_SOURCES) $(BENCHMARK_SOURCES)

# Object files
KERNEL_OBJECTS = $(KERNEL_SOURCES:$(SRC_DIR)/kernels/%.cu=$(BUILD_DIR)/kernels/%.o)
UTILS_OBJECTS = $(UTILS_SOURCES:$(SRC_DIR)/utils/%.cu=$(BUILD_DIR)/utils/%.o)
BENCHMARK_OBJECTS = $(BENCHMARK_SOURCES:$(SRC_DIR)/benchmark/%.cu=$(BUILD_DIR)/benchmark/%.o)

ALL_OBJECTS = $(KERNEL_OBJECTS) $(UTILS_OBJECTS) $(BENCHMARK_OBJECTS)

# Executables
MAIN_EXEC = $(BIN_DIR)/benchmark
TEST_EXEC = $(BIN_DIR)/test

# Default target
all: $(MAIN_EXEC)

# Create directories
$(BUILD_DIR)/kernels:
	mkdir -p $(BUILD_DIR)/kernels

$(BUILD_DIR)/utils:
	mkdir -p $(BUILD_DIR)/utils

$(BUILD_DIR)/benchmark:
	mkdir -p $(BUILD_DIR)/benchmark

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile kernel objects
$(BUILD_DIR)/kernels/%.o: $(SRC_DIR)/kernels/%.cu | $(BUILD_DIR)/kernels
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile utils objects
$(BUILD_DIR)/utils/%.o: $(SRC_DIR)/utils/%.cu | $(BUILD_DIR)/utils
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile benchmark objects
$(BUILD_DIR)/benchmark/%.o: $(SRC_DIR)/benchmark/%.cu | $(BUILD_DIR)/benchmark
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Main executable
$(MAIN_EXEC): $(ALL_OBJECTS) main.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) main.cu $(ALL_OBJECTS) $(LIBS) -o $@

# Test executable
$(TEST_EXEC): $(ALL_OBJECTS) test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) test.cu $(ALL_OBJECTS) $(LIBS) -o $@

# Test target
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run benchmark
run: $(MAIN_EXEC)
	./$(MAIN_EXEC)

# Profile with nvprof
profile: $(MAIN_EXEC)
	nvprof --metrics all ./$(MAIN_EXEC)

# Profile with nsight compute
profile-ncu: $(MAIN_EXEC)
	ncu --metrics all ./$(MAIN_EXEC)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(MAIN_EXEC)

.PHONY: all test clean run profile profile-ncu debug 
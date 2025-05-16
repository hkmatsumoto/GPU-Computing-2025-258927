cmake_minimum_required(VERSION 3.8)
project(spmv CUDA CXX)

# Set CUDA architecture for NVIDIA A30 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Add CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Create the main executable
add_executable(spmv src/main.cu)
target_compile_features(spmv PUBLIC cxx_std_11)

# Link against CUDA libraries
target_link_libraries(spmv PRIVATE cuda cudart)
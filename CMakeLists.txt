cmake_minimum_required(VERSION 3.21)
project(CudaGpibc CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaGpibc program.h rand_engine.h eval_gpu.h classify_test.cu classifier.h)

set_target_properties(CudaGpibc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

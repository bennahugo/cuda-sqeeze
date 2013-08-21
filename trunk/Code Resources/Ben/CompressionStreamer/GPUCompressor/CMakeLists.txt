CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
SET(LIB_TYPE STATIC)	#set the lib type
CUDA_ADD_LIBRARY(gpuCode ${LIB_TYPE} gpuCode.cu)



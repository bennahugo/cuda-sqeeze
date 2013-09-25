CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    #SET(CUDA_NVCC_FLAGS "-O3 -arch;sm_20 -Xcompiler -fopenmp")
    SET(CUDA_NVCC_FLAGS "-O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
SET(LIB_TYPE STATIC)	#set the lib type
CUDA_ADD_LIBRARY(gpuCode ${LIB_TYPE} gpuCode.cu)



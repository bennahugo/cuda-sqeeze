CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS "-arch;sm_20 -Xcompiler -fopenmp")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
SET(LIB_TYPE STATIC)	#set the lib type
CUDA_ADD_LIBRARY(gpuCode ${LIB_TYPE} gpuCode.cu)



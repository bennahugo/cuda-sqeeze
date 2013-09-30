project(compressionstreamer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
#declare executable
find_package(CUDA)
cuda_add_executable(compressionstreamer AstroReader/stride.cpp AstroReader/file.cpp AstroReader/stridefactory.cpp timer.cpp main.cpp)
install(TARGETS compressionstreamer RUNTIME DESTINATION bin)
#link internal libraries
add_subdirectory(Compressor)
add_subdirectory(GPUCompressor)
target_link_libraries(compressionstreamer cpuCode)
target_link_libraries(compressionstreamer gpuCode)
#link external libraries
SET(CMAKE_REQUIRED_FLAGS "-Wl,--no-undefined")
find_package(hdf5)
target_link_libraries(compressionstreamer hdf5)

find_package(rt)
target_link_libraries(compressionstreamer rt)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
target_link_libraries(compressionstreamer gomp)

set(CMAKE_BUILD_TYPE Release)
set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp -fno-stack-protector" )
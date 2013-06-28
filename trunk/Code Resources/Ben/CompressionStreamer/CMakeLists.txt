project(compressionstreamer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

add_executable(compressionstreamer Compressor/cpuCode.cpp AstroReader/stride.cpp AstroReader/file.cpp AstroReader/stridefactory.cpp Timer.cpp main.cpp)

install(TARGETS compressionstreamer RUNTIME DESTINATION bin)

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
add_subdirectory(AstroReader)
add_subdirectory(Compressor)
set(CMAKE_BUILD_TYPE Release)
set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp -mfpmath=sse -mmmx -msse -msse2" )
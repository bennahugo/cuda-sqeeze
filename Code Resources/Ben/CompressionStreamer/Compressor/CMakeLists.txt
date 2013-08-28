CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Release)
#enable this if you need SSE AND XOP support!!!:
#set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp -mfpmath=sse -mmmx -msse -msse2 -msse3 -msse4 -msse4a -msse4.1 -msse4.2 -mxop" )
set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp" )
add_library(cpuCode cpuCode.cpp)

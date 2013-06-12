project(compressionstreamer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

add_executable(compressionstreamer AstroReader/stride.cpp AstroReader/file.cpp AstroReader/stridefactory.cpp main.cpp)

install(TARGETS compressionstreamer RUNTIME DESTINATION bin)

SET(CMAKE_REQUIRED_FLAGS "-Wl,--no-undefined")
find_package(hdf5)

target_link_libraries(compressionstreamer hdf5)
add_subdirectory(AstroReader)
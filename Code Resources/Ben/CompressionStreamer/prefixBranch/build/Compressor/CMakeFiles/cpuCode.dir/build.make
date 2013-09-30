# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bhugo/projects/CompressionStreamer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bhugo/projects/CompressionStreamer/build

# Include any dependencies generated for this target.
include Compressor/CMakeFiles/cpuCode.dir/depend.make

# Include the progress variables for this target.
include Compressor/CMakeFiles/cpuCode.dir/progress.make

# Include the compile flags for this target's objects.
include Compressor/CMakeFiles/cpuCode.dir/flags.make

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o: Compressor/CMakeFiles/cpuCode.dir/flags.make
Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o: ../Compressor/cpuCode.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/bhugo/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o"
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cpuCode.dir/cpuCode.cpp.o -c /home/bhugo/projects/CompressionStreamer/Compressor/cpuCode.cpp

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpuCode.dir/cpuCode.cpp.i"
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/bhugo/projects/CompressionStreamer/Compressor/cpuCode.cpp > CMakeFiles/cpuCode.dir/cpuCode.cpp.i

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpuCode.dir/cpuCode.cpp.s"
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/bhugo/projects/CompressionStreamer/Compressor/cpuCode.cpp -o CMakeFiles/cpuCode.dir/cpuCode.cpp.s

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.requires:
.PHONY : Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.requires

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.provides: Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.requires
	$(MAKE) -f Compressor/CMakeFiles/cpuCode.dir/build.make Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.provides.build
.PHONY : Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.provides

Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.provides.build: Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o

# Object files for target cpuCode
cpuCode_OBJECTS = \
"CMakeFiles/cpuCode.dir/cpuCode.cpp.o"

# External object files for target cpuCode
cpuCode_EXTERNAL_OBJECTS =

Compressor/libcpuCode.a: Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o
Compressor/libcpuCode.a: Compressor/CMakeFiles/cpuCode.dir/build.make
Compressor/libcpuCode.a: Compressor/CMakeFiles/cpuCode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libcpuCode.a"
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && $(CMAKE_COMMAND) -P CMakeFiles/cpuCode.dir/cmake_clean_target.cmake
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpuCode.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Compressor/CMakeFiles/cpuCode.dir/build: Compressor/libcpuCode.a
.PHONY : Compressor/CMakeFiles/cpuCode.dir/build

Compressor/CMakeFiles/cpuCode.dir/requires: Compressor/CMakeFiles/cpuCode.dir/cpuCode.cpp.o.requires
.PHONY : Compressor/CMakeFiles/cpuCode.dir/requires

Compressor/CMakeFiles/cpuCode.dir/clean:
	cd /home/bhugo/projects/CompressionStreamer/build/Compressor && $(CMAKE_COMMAND) -P CMakeFiles/cpuCode.dir/cmake_clean.cmake
.PHONY : Compressor/CMakeFiles/cpuCode.dir/clean

Compressor/CMakeFiles/cpuCode.dir/depend:
	cd /home/bhugo/projects/CompressionStreamer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bhugo/projects/CompressionStreamer /home/bhugo/projects/CompressionStreamer/Compressor /home/bhugo/projects/CompressionStreamer/build /home/bhugo/projects/CompressionStreamer/build/Compressor /home/bhugo/projects/CompressionStreamer/build/Compressor/CMakeFiles/cpuCode.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Compressor/CMakeFiles/cpuCode.dir/depend


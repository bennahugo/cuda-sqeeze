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
CMAKE_SOURCE_DIR = /home/benjamin/projects/CompressionStreamer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/benjamin/projects/CompressionStreamer/build

# Include any dependencies generated for this target.
include CMakeFiles/compressionstreamer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/compressionstreamer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/compressionstreamer.dir/flags.make

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o: ../Compressor/cpuCode.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o -c /home/benjamin/projects/CompressionStreamer/Compressor/cpuCode.cpp

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/Compressor/cpuCode.cpp > CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.i

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/Compressor/cpuCode.cpp -o CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.s

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.requires

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.provides: CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.provides

CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o: ../AstroReader/stride.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o -c /home/benjamin/projects/CompressionStreamer/AstroReader/stride.cpp

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/AstroReader/stride.cpp > CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.i

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/AstroReader/stride.cpp -o CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.s

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.requires

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.provides: CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.provides

CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o: ../AstroReader/file.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o -c /home/benjamin/projects/CompressionStreamer/AstroReader/file.cpp

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/AstroReader/file.cpp > CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.i

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/AstroReader/file.cpp -o CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.s

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.requires

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.provides: CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.provides

CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o: ../AstroReader/stridefactory.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o -c /home/benjamin/projects/CompressionStreamer/AstroReader/stridefactory.cpp

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/AstroReader/stridefactory.cpp > CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.i

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/AstroReader/stridefactory.cpp -o CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.s

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.requires

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.provides: CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.provides

CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o

CMakeFiles/compressionstreamer.dir/Timer.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/Timer.cpp.o: ../Timer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/Timer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/Timer.cpp.o -c /home/benjamin/projects/CompressionStreamer/Timer.cpp

CMakeFiles/compressionstreamer.dir/Timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/Timer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/Timer.cpp > CMakeFiles/compressionstreamer.dir/Timer.cpp.i

CMakeFiles/compressionstreamer.dir/Timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/Timer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/Timer.cpp -o CMakeFiles/compressionstreamer.dir/Timer.cpp.s

CMakeFiles/compressionstreamer.dir/Timer.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/Timer.cpp.o.requires

CMakeFiles/compressionstreamer.dir/Timer.cpp.o.provides: CMakeFiles/compressionstreamer.dir/Timer.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/Timer.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/Timer.cpp.o.provides

CMakeFiles/compressionstreamer.dir/Timer.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/Timer.cpp.o

CMakeFiles/compressionstreamer.dir/main.cpp.o: CMakeFiles/compressionstreamer.dir/flags.make
CMakeFiles/compressionstreamer.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/projects/CompressionStreamer/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compressionstreamer.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compressionstreamer.dir/main.cpp.o -c /home/benjamin/projects/CompressionStreamer/main.cpp

CMakeFiles/compressionstreamer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compressionstreamer.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/projects/CompressionStreamer/main.cpp > CMakeFiles/compressionstreamer.dir/main.cpp.i

CMakeFiles/compressionstreamer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compressionstreamer.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/projects/CompressionStreamer/main.cpp -o CMakeFiles/compressionstreamer.dir/main.cpp.s

CMakeFiles/compressionstreamer.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/compressionstreamer.dir/main.cpp.o.requires

CMakeFiles/compressionstreamer.dir/main.cpp.o.provides: CMakeFiles/compressionstreamer.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/compressionstreamer.dir/build.make CMakeFiles/compressionstreamer.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/compressionstreamer.dir/main.cpp.o.provides

CMakeFiles/compressionstreamer.dir/main.cpp.o.provides.build: CMakeFiles/compressionstreamer.dir/main.cpp.o

# Object files for target compressionstreamer
compressionstreamer_OBJECTS = \
"CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o" \
"CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o" \
"CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o" \
"CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o" \
"CMakeFiles/compressionstreamer.dir/Timer.cpp.o" \
"CMakeFiles/compressionstreamer.dir/main.cpp.o"

# External object files for target compressionstreamer
compressionstreamer_EXTERNAL_OBJECTS =

compressionstreamer: CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/Timer.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/main.cpp.o
compressionstreamer: CMakeFiles/compressionstreamer.dir/build.make
compressionstreamer: CMakeFiles/compressionstreamer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable compressionstreamer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compressionstreamer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/compressionstreamer.dir/build: compressionstreamer
.PHONY : CMakeFiles/compressionstreamer.dir/build

CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/Compressor/cpuCode.cpp.o.requires
CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/AstroReader/stride.cpp.o.requires
CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/AstroReader/file.cpp.o.requires
CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/AstroReader/stridefactory.cpp.o.requires
CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/Timer.cpp.o.requires
CMakeFiles/compressionstreamer.dir/requires: CMakeFiles/compressionstreamer.dir/main.cpp.o.requires
.PHONY : CMakeFiles/compressionstreamer.dir/requires

CMakeFiles/compressionstreamer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/compressionstreamer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/compressionstreamer.dir/clean

CMakeFiles/compressionstreamer.dir/depend:
	cd /home/benjamin/projects/CompressionStreamer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/benjamin/projects/CompressionStreamer /home/benjamin/projects/CompressionStreamer /home/benjamin/projects/CompressionStreamer/build /home/benjamin/projects/CompressionStreamer/build /home/benjamin/projects/CompressionStreamer/build/CMakeFiles/compressionstreamer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/compressionstreamer.dir/depend


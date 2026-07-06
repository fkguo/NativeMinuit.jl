# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Build the NativeMinuit C++ reference-data harness.
#
# Uses add_subdirectory against the pinned C++ Minuit2 standalone at
# ../reference/Minuit2_cpp (GooFit/Minuit2 @ 57dc936, v6.24.0).
#
# Usage:
#   cd tools && mkdir -p build && cd build
#   cmake ..
#   make
#   ./cpp_trace_harness ../../test/reference_data/

cmake_minimum_required(VERSION 3.15)
project(nativeminuit_reference_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Minuit2 standalone's CMakeLists.txt declares cmake_minimum_required(VERSION 3.1)
# which CMake 4.x refuses without an explicit policy override. Set the
# minimum-policy-version cache var so add_subdirectory works.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# Build Minuit2 standalone as a subdirectory; do not install
# (we only need the linkable target).
set(BUILD_TESTING OFF CACHE BOOL "Do not build Minuit2 tests" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../reference/Minuit2_cpp
                 ${CMAKE_BINARY_DIR}/minuit2_build
                 EXCLUDE_FROM_ALL)

add_executable(cpp_trace_harness cpp_trace_harness.cxx)
target_link_libraries(cpp_trace_harness PUBLIC Minuit2::Minuit2)

# Convenience: a target that runs the harness with the default output dir.
add_custom_target(regen_reference
    COMMAND $<TARGET_FILE:cpp_trace_harness>
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/reference_data
    DEPENDS cpp_trace_harness
    COMMENT "Regenerating test/reference_data/*.json"
)

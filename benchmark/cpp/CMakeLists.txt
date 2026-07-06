# SPDX-License-Identifier: LGPL-2.1-or-later
#
# C++ Minuit2 wall-time benchmark — Phase 0 §3.4 Criterion 2 reference.
#
# Runs the §3.3 corpus N times and reports median per-call wall time
# to stdout as JSON. Consumed by benchmark/compare_cpp.jl which pulls
# the corresponding Julia times from
# benchmark/.julia-perf/runs/latest/benchmarks.json.

cmake_minimum_required(VERSION 3.15)
project(nativeminuit_cpp_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use optimized build by default for fair comparison.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Skip Minuit2 tests" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../reference/Minuit2_cpp
                 ${CMAKE_BINARY_DIR}/minuit2_build
                 EXCLUDE_FROM_ALL)

add_executable(cpp_bench cpp_bench.cxx)
target_link_libraries(cpp_bench PUBLIC Minuit2::Minuit2)

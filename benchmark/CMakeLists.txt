cmake_minimum_required(VERSION 3.15)

#
# Project details
#

project(
  ${CMAKE_PROJECT_NAME}Benchmarks
  LANGUAGES CXX
)

verbose_message("Adding tests under ${CMAKE_PROJECT_NAME}Benchmarks...")

#
# Set the sources for the unit tests and add the executable(s)
#

set(test_sources 
  src/main_benchmark.cpp
)
add_executable(${PROJECT_NAME} ${test_sources})

#
# Set the compiler standard
#

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

#
# Setup code coverage if enabled
#

if (${CMAKE_PROJECT_NAME}_ENABLE_CODE_COVERAGE)
  target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(${CMAKE_PROJECT_NAME} PUBLIC -fprofile-arcs -ftest-coverage)
  verbose_message("Code coverage is enabled and provided with GCC.")
endif()

#
# Load the desired unit testing framework
#
# Currently supported: GoogleTest (and GoogleMock), Catch2.

if(${CMAKE_PROJECT_NAME}_BUILD_EXECUTABLE)
	set(${CMAKE_PROJECT_NAME}_TEST_LIB ${CMAKE_PROJECT_NAME}_LIB)
else()
	set(${CMAKE_PROJECT_NAME}_TEST_LIB ${CMAKE_PROJECT_NAME})
endif()
    
find_package(benchmark REQUIRED)

if(${CMAKE_PROJECT_NAME}_LIB)
  target_link_libraries(
    ${PROJECT_NAME} 
    PUBLIC
      benchmark::benchmark
      spdlog::spdlog
      ${${CMAKE_PROJECT_NAME}_TEST_LIB}
  )
else()
  target_link_libraries(
    ${PROJECT_NAME} 
    PUBLIC
      benchmark::benchmark
      spdlog::spdlog
  )
endif()
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(colorizeLaser)

find_package(PCL 1.0 REQUIRED)
find_package(OpenMP)
if(OpenMP_FOUND)
	link_directories(${OpenMP_LIBRARY_DIRS})
	include_directories(${OpenMP_INCLUDE_DIRS})
endif(OpenMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable( colorizeLaser colorizeLaser.cpp )
target_link_libraries( colorizeLaser ${PCL_LIBRARIES} )

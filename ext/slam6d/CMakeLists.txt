cmake_minimum_required(VERSION 2.8)

set(SLAM_SOURCES
	Boctree.cc
	allocator.cc
	point_type.cc
	searchTree.cc
	scan_io_oct.cc
)
SET_SOURCE_FILES_PROPERTIES(${SLAM_SOURCES} PROPERTIES COMPILE_FLAGS "-std=gnu++0x")

ADD_LIBRARY(slam6d_static STATIC ${SLAM_SOURCES})
set_target_properties(slam6d_static PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

ADD_LIBRARY(slam6d SHARED ${SLAM_SOURCES})
set_target_properties(slam6d PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


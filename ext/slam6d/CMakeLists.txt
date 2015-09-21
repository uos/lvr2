cmake_minimum_required(VERSION 2.8)

include_directories(include)

set(SLAM_SOURCES
	Boctree.cc
	allocator.cc
	point_type.cc
	searchTree.cc
	scan_io_oct.cc
)
set_source_files_properties(${SLAM_SOURCES} PROPERTIES COMPILE_FLAGS "-std=gnu++0x")

add_library(lvrslam6d_static STATIC ${SLAM_SOURCES})
set_target_properties(lvrslam6d_static PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_library(lvrslam6d SHARED ${SLAM_SOURCES})
set_target_properties(lvrslam6d PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

install(TARGETS lvrslam6d_static lvrslam6d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

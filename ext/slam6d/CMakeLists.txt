cmake_minimum_required(VERSION 2.8)

find_path(SLAM6D_INCLUDE_DIR slam6d/scan_io.h PATH include)

include_directories(${SLAM6D_INCLUDE_DIR})

set(SLAM_SOURCES
	Boctree.cc
	allocator.cc
	point_type.cc
	searchTree.cc
	scan_io_oct.cc
)
set_source_files_properties(${SLAM_SOURCES} PROPERTIES COMPILE_FLAGS "-std=gnu++0x")

add_library(lvrslam6d_static STATIC ${SLAM_SOURCES})

add_library(lvrslam6d SHARED ${SLAM_SOURCES})

# this intentionally installs to lvr/ext/slam6d/slam6d to keep a separate include path
install(DIRECTORY include/slam6d DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/slam6d)

install(TARGETS lvrslam6d_static lvrslam6d
  EXPORT lvrTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/slam6d)

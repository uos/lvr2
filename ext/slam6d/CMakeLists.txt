cmake_minimum_required(VERSION 2.8)

# set include dirs
set(SLAM6D_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/slam6d CACHE PATH "install location of slam6d headers")
find_path(SLAM6D_INCLUDE_DIR slam6d/scan_io.h PATH include)
mark_as_advanced(SLAM6D_INCLUDE_DIR SLAM6D_INSTALL_INCLUDE_DIR)

include_directories(${SLAM6D_INCLUDE_DIR})

set(SLAM_SOURCES
	Boctree.cc
	allocator.cc
	point_type.cc
	searchTree.cc
	scan_io_oct.cc
)
set_source_files_properties(${SLAM_SOURCES} PROPERTIES COMPILE_FLAGS "-std=gnu++0x")

add_library(lvr2slam6d_static STATIC ${SLAM_SOURCES})

add_library(lvr2slam6d SHARED ${SLAM_SOURCES})

# this intentionally installs to lvr2/ext/slam6d/slam6d to keep a separate include path
install(DIRECTORY include/slam6d DESTINATION ${SLAM6D_INSTALL_INCLUDE_DIR})

install(TARGETS lvr2slam6d_static lvr2slam6d
  EXPORT lvr2Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

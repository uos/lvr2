cmake_minimum_required(VERSION 3.4...3.16)


# set include dirs
set(NANOFLANN_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/nanoflann)
find_path(NANOFLANN_INCLUDE_DIR nanoflann.hpp PATH .)
mark_as_advanced(NANOFLANN_INCLUDE_DIR NANOFLANN_INSTALL_INCLUDE_DIR)

message(STATUS "installing nanoflann to -> ${NANOFLANN_INSTALL_INCLUDE_DIR}")

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/nanoflann
  FILES_MATCHING PATTERN "*.hpp")

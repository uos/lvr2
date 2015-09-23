cmake_minimum_required(VERSION 2.8)

project(nanoflann)

find_path(NANOFLANN_INCLUDE_DIR nanoflann.hpp PATH .)

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/nanoflann
  FILES_MATCHING PATTERN "*.hpp")

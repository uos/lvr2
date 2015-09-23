cmake_minimum_required(VERSION 2.8)

project(psimpl)

find_path(PSIMPL_INCLUDE_DIR psimpl.h PATH .)

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/psimpl
  FILES_MATCHING PATTERN "*.h")

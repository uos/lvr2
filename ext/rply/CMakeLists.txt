cmake_minimum_required(VERSION 2.8)

project(rply)

find_path(RPLY_INCLUDE_DIR rply.h PATH .)

add_library(lvrrply_static STATIC rply.c)

add_library(lvrrply SHARED rply.c)

install(TARGETS lvrrply_static lvrrply
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/rply
  FILES_MATCHING PATTERN "*.h")

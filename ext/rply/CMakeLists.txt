cmake_minimum_required(VERSION 2.8)

#project(rply)

# set include dirs
set(RPLY_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/rply CACHE PATH "install location of rply library header")
find_path(RPLY_INCLUDE_DIR rply.h PATH .)
mark_as_advanced(RPLY_INCLUDE_DIR RPLY_INSTALL_INCLUDE_DIR)

include_directories(${RPLY_INCLUDE_DIR})

add_library(lvr2rply_static STATIC rply.c)

add_library(lvr2rply SHARED rply.c)

install(DIRECTORY . DESTINATION ${RPLY_INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h")

install(TARGETS lvr2rply_static lvr2rply
  EXPORT lvr2Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

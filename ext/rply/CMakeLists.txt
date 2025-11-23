cmake_minimum_required(VERSION 3.4...3.16)

#project(rply)

# set include dirs
set(RPLY_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/rply)
find_path(RPLY_INCLUDE_DIR rply.h PATH .)
mark_as_advanced(RPLY_INCLUDE_DIR RPLY_INSTALL_INCLUDE_DIR)

include_directories(${RPLY_INCLUDE_DIR})

add_library(lvr2rply_static STATIC rply.c)

add_library(lvr2rply SHARED rply.c)

set_target_properties(lvr2rply
  PROPERTIES
    SOVERSION ${lvr2_VERSION_MAJOR}
    VERSION ${lvr2_VERSION}
)

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/rply
  FILES_MATCHING PATTERN "*.h")

install(TARGETS lvr2rply_static lvr2rply
  EXPORT lvr2-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

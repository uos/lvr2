cmake_minimum_required(VERSION 2.8)

set(CTPL_INSTALL_INCLUDE_DIR
    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/CTPL
    CACHE
    PATH
    "install location of CTPL library header")

find_path(CTPL_INCLUDE_DIR ctpl.h PATH .)
mark_as_advanced(CTPL_INCLUDE_DIR CTPL_INSTALL_INCLUDE_DIR)

include_directories(${CTPL_INCLUDE_DIR})

install(DIRECTORY . DESTINATION ${CTPL_INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h")

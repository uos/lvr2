cmake_minimum_required(VERSION 2.8)

# set include dirs
set(PSIMPL_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/psimpl CACHE PATH "install location of psimple header")
find_path(PSIMPL_INCLUDE_DIR psimpl.h PATH .)
mark_as_advanced(PSIMPL_INCLUDE_DIR PSIMPL_INSTALL_INCLUDE_DIR)

install(DIRECTORY . DESTINATION ${PSIMPL_INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h")

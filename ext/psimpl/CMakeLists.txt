cmake_minimum_required(VERSION 3.4...3.16)

# set include dirs
set(PSIMPL_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/psimpl)
find_path(PSIMPL_INCLUDE_DIR psimpl.h PATH .)

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr2/ext/psimpl
  FILES_MATCHING PATTERN "*.h")

###############################################################################
# Find Las Vegas Reconstruction Kit
#
# This sets the following variables:
# LSSR_FOUND - True if LSSR was found.
# LSSR_INCLUDE_DIRS - Directories containing the LSSR include files.
# LSSR_LIBRARIES - Libraries needed to use LSSR.
# 

find_package(PkgConfig QUIET)

#add a hint so that it can find it without the pkg-config
find_path(LSSR_INCLUDE_DIR MeshGenerator.hpp
          HINTS  ~/meshing.tigelbri/include/liblvr/reconstruction)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LSSR DEFAULT_MSG LSSR_INCLUDE_DIR)



 
if(LSSR_FOUND)  
  # Add the include directories
  set(LSSR_INCLUDE_DIRS "${LSSR_INCLUDE_DIR}/.." "${LSSR_INCLUDE_DIR}/../../../ext/rply" "${LSSR_INCLUDE_DIR}/../../../ext/" "${LSSR_INCLUDE_DIR}/../../../ext/stann" "${LSSR_INCLUDE_DIR}/../../../ext/psimpl/")  
  set(LSSR_LIBRARIES "${LSSR_INCLUDE_DIR}/../../../build/src/liblvr/" "${LSSR_INCLUDE_DIR}/../../../build/ext/laslib/" "${LSSR_INCLUDE_DIR}/../../../build/ext/rply/" "${LSSR_INCLUDE_DIR}/../../../build/ext/slam6d/" "${LSSR_INCLUDE_DIR}/../../../lib/")
  message(STATUS "LSSR found (include: ${LSSR_INCLUDE_DIRS}, libraries: ${LSSR_LIBRARIES})")
endif(LSSR_FOUND)


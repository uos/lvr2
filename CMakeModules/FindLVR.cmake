###############################################################################
# Find Las Vegas Reconstruction Kit
#
# This sets the following variables:
# LSSR_FOUND - True if LVR was found.
# LVR_INCLUDE_DIRS - Directories containing the LVR include files.
# LVR_LIBRARIES - Libraries needed to use LVR.
# 

find_package(PkgConfig QUIET)

#add a hint so that it can find it without the pkg-config
find_path(LVR_INCLUDE_DIR MeshGenerator.hpp
          HINTS  /usr/local/include/lvr/reconstruction)

message(STATUS "LVR include directory: ${LVR_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LVR DEFAULT_MSG LVR_INCLUDE_DIR)



 
if(LVR_FOUND)  
  # Add the include directories
  set(LVR_INCLUDE_DIRS "${LVR_INCLUDE_DIR}/.." "${LVR_INCLUDE_DIR}/../../../ext/rply" "${LVR_INCLUDE_DIR}/../../../ext/" "${LVR_INCLUDE_DIR}/../../../ext/stann" "${LVR_INCLUDE_DIR}/../../../ext/psimpl/")  
  set(LVR_LIBRARIES "${LVR_INCLUDE_DIR}/../../../build/src/liblvr/" "${LVR_INCLUDE_DIR}/../../../build/ext/laslib/" "${LVR_INCLUDE_DIR}/../../../build/ext/rply/" "${LVR_INCLUDE_DIR}/../../../build/ext/slam6d/" "${LVR_INCLUDE_DIR}/../../../lib/")
  message(STATUS "LVR found (include: ${LVR_INCLUDE_DIRS}, libraries: ${LVR_LIBRARIES})")
endif(LVR_FOUND)


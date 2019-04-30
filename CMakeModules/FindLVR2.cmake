###############################################################################
# Find Las Vegas Reconstruction Kit
#
# This sets the following variables:
# LVR2_FOUND - True if LVR2 was found.
# LVR2_INCLUDE_DIRS - Directories containing the LVR2 include files.
# LVR2_LIBRARIES - Libraries needed to use LVR2.
# 

find_package(PkgConfig QUIET)

#add a hint so that it can find it without the pkg-config
find_path(LVR2_INCLUDE_DIR MeshGenerator.hpp
          HINTS  /usr/local/include/lvr2/reconstruction)

message(STATUS "LVR2 include directory: ${LVR2_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LVR2 DEFAULT_MSG LVR2_INCLUDE_DIR)



 
if(LVR2_FOUND)  
  # Add the include directories
  set(LVR2_INCLUDE_DIRS "${LVR2_INCLUDE_DIR}/.." "${LVR2_INCLUDE_DIR}/../../../ext/rply" "${LVR2_INCLUDE_DIR}/../../../ext/" "${LVR2_INCLUDE2_DIR}/../../../ext/stann" "${LVR2_INCLUDE_DIR}/../../../ext/psimpl/")
  set(LVR2_LIBRARIES "${LVR2_INCLUDE_DIR}/../../../build/src/liblvr2/" "${LVR2_INCLUDE_DIR}/../../../build/ext/laslib/" "${LVR2_INCLUDE_DIR}/../../../build/ext/rply/" "${LVR2_INCLUDE_DIR}/../../../build/ext/slam6d/" "${LVR2_INCLUDE_DIR}/../../../lib/")
  message(STATUS "LVR2 found (include: ${LVR2_INCLUDE_DIRS}, libraries: ${LVR2_LIBRARIES})")
endif(LVR2_FOUND)


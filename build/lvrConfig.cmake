# -----------------------------------------------------------------
# lvr's cmake configuration
#
# provided options:
# - lvr_USE_STATIC_LIBS(=OFF) to link the static libraries of lvr
#
# provided interface variables:
# - lvr_INCLUDE_DIRS
# - lvr_DEFINITIONS
# - lvr_LIBRARIES
#
# Do not forget to add_defintions(${lvr_DEFINITIONS}) as they
# describe the build configuration of liblvr.
#
# -----------------------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/lvrTargets.cmake")

cmake_policy(PUSH)
cmake_policy(SET CMP0012 NEW)

set(lvr_INCLUDE_DIRS /usr/local/include/lvr/ext/nanoflann;/usr/local/include/lvr/ext/psimpl;/usr/local/include/lvr/ext/rply;/usr/local/include/lvr/ext/laslib;/usr/local/include/lvr/ext/slam6d;/usr/local/include)
set(lvr_DEFINITIONS -DLVR_USE_OPEN_MP;-DLVR_USE_STANN)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)
list(APPEND lvr_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

find_package(VTK REQUIRED)
list(APPEND lvr_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})
list(APPEND lvr_DEFINITIONS ${VTK_DEFINTIONS})

# nabo
if(FALSE)
  list(APPEND lvr_INCLUDE_DIRS )
endif()

# pcl
if(0)
  find_package(PCL REQUIRED)
  list(APPEND lvr_INCLUDE_DIRS ${PCL_INCLUDE_DIRS})
  list(APPEND lvr_DEFINITIONS ${PCL_DEFINITIONS})
endif()

# cgal
if(0)
  find_package(CGAL REQUIRED)
  list(APPEND lvr_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS})
endif()

find_package(Boost REQUIRED)
list(APPEND lvr_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
list(APPEND lvr_DEFINITIONS ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

find_package(OpenGL REQUIRED)
list(APPEND lvr_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})

find_package(GLUT REQUIRED)
list(APPEND lvr_INCLUDE_DIRS ${GLUT_INCLUDE_DIR})

# libfreenect
if()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBFREENECT REQUIRED libfreenect)
  list(APPEND lvr_INCLUDE_DIRS ${LIBFREENECT_INCLUDE_DIRS})
endif()

# stann
if(TRUE)
  list(APPEND lvr_INCLUDE_DIRS /home/student/i/imitschke/local/include/STANN)
endif()

cmake_policy(POP)

option(lvr_USE_STATIC_LIBS OFF)
if(lvr_USE_STATIC_LIBS)
  set(lvr_LIBRARIES lvr::lvr_static)
else()
  set(lvr_LIBRARIES lvr::lvr)
endif()

set(lvr_FOUND TRUE)

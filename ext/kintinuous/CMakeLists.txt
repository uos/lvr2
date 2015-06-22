cmake_minimum_required(VERSION 2.8.8)

# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
if (DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# ---[ Solution name
project(kfusion C CXX)

# ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

# -- [ this option causes problems for some reason, so turn it off before searching for CUDA
if(MSVC)
	# Setting this to true brakes Visual Studio builds.
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif()

# ---[ find dependencies
find_package(OpenCV 2.4.8 REQUIRED COMPONENTS core viz highgui imgproc features2d nonfree )
find_package(CUDA 5.0 REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(LSSR REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    message(STATUS "-- Found Boost package" )
else(Boost_FOUND)
    message( SEND_ERROR "-- could not find Boost package" )
endif(Boost_FOUND)

if(OpenGL_FOUND)
    link_directories(${OpenGL_LIBRARY_DIRS})
    include_directories(${OpenGL_INCLUDE_DIRS})
    message( "OPEN GL LIBS 1 : " ${OPENGL_LIBRARIES}) 
    message( "OPEN GL LIBS 2 : " ${OPENGL_LIBRARY}) 
endif(OpenGL_FOUND)

if ( GLUT_FOUND )
    message(STATUS "-- Found OpenGL Utility Toolkit" )
    include_directories(${GLUT_INCLUDE_DIR})
else ( GLUT_FOUND )
    message( SEND_ERROR "-- could not find OpenGL Utility Toolkit" )
endif ( GLUT_FOUND )

if ( EIGEN3_FOUND )
    message(STATUS "-- Found Eigen3" )
    include_directories(${EIGEN3_INCLUDE_DIR})
else ( EIGEN3_FOUND )
    message( SEND_ERROR "-- could not find Eigen3" )
endif ( EIGEN3_FOUND )

if(LINUX)
if ( GLUT_Xmu_LIBRARY )
    message( "-- Found X.Org X11 libXmu/libXmuu runtime libraries" )
else ( GLUT_Xmu_LIBRARY )
    message( SEND_ERROR "-- Could not find X.Org X11 libXmu/libXmuu runtime libraries" )
endif ( GLUT_Xmu_LIBRARY )

if ( GLUT_Xi_LIBRARY )
    message( "-- Found X.Org X11 libXi runtime library" )
else ( GLUT_Xi_LIBRARY )
    message( SEND_ERROR "-- Could not find X.Org X11 libXi runtime library" )
endif ( GLUT_Xi_LIBRARY )

if ( OPENGL_glu_LIBRARY ) 
    message( "-- Found GLU library" )
else ( OPENGL_glu_LIBRARY ) 
    message( SEND_ERROR "-- Could not find GLU library" )
endif ( OPENGL_glu_LIBRARY ) 

endif(LINUX)

add_definitions(-std=c++11 -D_GLIBCXX_PARALLEL)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENNI2_INCLUDE_DIRS} ${LSSR_INCLUDE_DIRS})

# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ cuda settings
set(HAVE_CUDA 1)
#list(APPEND CUDA_NVCC_FLAGS "-arch;compute_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35 ")

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

warnings_disable(CMAKE_CXX_FLAGS /wd4985)

add_subdirectory(kfusion)
add_subdirectory(apps)

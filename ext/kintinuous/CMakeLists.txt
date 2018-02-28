cmake_minimum_required(VERSION 2.8.8)
project(kfusion C CXX)

find_package(OpenNI2 REQUIRED)
include_directories(OPENNI2_INCLUDE_DIRS)

##############################################################
# INCLUDE DIRECTORIES
##############################################################

include_directories(kfusion/include)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENNI2_INCLUDE_DIRS})

##############################################################
# COMPILER SPECIFIC SETTINGS
##############################################################

# This option causes problems for some reason, so turn it off before searching for CUDA
if(MSVC)
    # Setting this to true brakes Visual Studio builds.
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif()

##############################################################
# CUDA SETTINGS
##############################################################

set(HAVE_CUDA 1)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=sm_50;")

##############################################################
# CHECK FOR OPENCV VIZ MODULE
##############################################################

find_package(OpenCV 2.4.8 COMPONENTS core viz highgui imgproc features2d nonfree)
if(NOT OPENCV_VIZ_FOUND)
message("WARNING: Could not find OpenCV viz module. Kinfu App will not be built.")
else()
    add_subdirectory(kfusion)
    #add_subdirectory(qt_app)
endif()

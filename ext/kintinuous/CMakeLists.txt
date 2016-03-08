
find_package(OpenNI2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENNI2_INCLUDE_DIRS})

# -- [ this option causes problems for some reason, so turn it off before searching for CUDA
if(MSVC)
	# Setting this to true brakes Visual Studio builds.
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif()

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35")

list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50")

if(UNIX OR APPLE)
  #list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC")
endif()

#warnings_disable(CMAKE_CXX_FLAGS /wd4985)
add_subdirectory(kfusion)


##############################################################
# VTK use file include. Has to be called after all CUDA code
# was generated and compiled. The VTK definitions corrupt the 
# nvcc call
##############################################################

include(${VTK_USE_FILE})

if(OPENCV_VIZ_FOUND)
add_subdirectory(apps)
endif()

add_subdirectory(qt_app)

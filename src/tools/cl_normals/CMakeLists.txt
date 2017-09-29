#####################################################################################
# Set source files
#####################################################################################

find_package(OpenCL REQUIRED)
include_directories(OpenCL_INCLUDE_DIRS)
message(STATUS "Loaded OpenCL version ${OpenCL_VERSION_STRING}")
message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")

set(LVR_CL_NORMAL_SRC
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_CL_NORMAL_DEPS
	lvr
	${OpenCL_LIBRARIES}
	)



#####################################################################################
# Add executable
#####################################################################################
add_executable(lvr_cl_normals ${LVR_CL_NORMAL_SRC})

target_link_libraries(lvr_cl_normals ${LVR_CL_NORMAL_DEPS})

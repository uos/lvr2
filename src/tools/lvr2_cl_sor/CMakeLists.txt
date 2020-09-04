#####################################################################################
# Set source files
#####################################################################################

# find_package(OPENCL REQUIRED)
# include_directories(OpenCL_INCLUDE_DIRS)
# message(STATUS "Loaded OpenCL version ${OpenCL_VERSION_STRING}")
# message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")

set(LVR2_CL_NORMAL_SRC
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CL_NORMAL_DEPS
    lvr2_static
    ${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################
add_executable(lvr2_cl_sor ${LVR2_CL_NORMAL_SRC})
target_link_libraries(lvr2_cl_sor ${LVR2_CL_NORMAL_DEPS})

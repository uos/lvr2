#####################################################################################
# Package includes
#####################################################################################

find_package(OpenCL REQUIRED)
include_directories(OpenCL_INCLUDE_DIRS)
message(STATUS "Loaded OpenCL version ${OpenCL_VERSION_STRING}")
message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")


#####################################################################################
# Add executable
#####################################################################################
add_executable(lvr2_raycaster_test
    Main.cpp
)

target_link_libraries(lvr2_raycaster_test
    lvr2_static
    ${OpenCL_LIBRARIES}
)

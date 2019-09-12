#####################################################################################
# Package includes
#####################################################################################

find_package(OPENCL)
if(OPENCL_FOUND)

#####################################################################################
# Add executable
#####################################################################################
add_executable(lvr2_example_raycaster_test
    Main.cpp
)

target_link_libraries(lvr2_example_raycaster_test
    lvr2_static
    ${OpenCL_LIBRARIES}
)
else()
message(WARNING "Could not compile lvr2_example_raycaster_test. OpenCL not found.")
endif()
#####################################################################################
# Package includes
#####################################################################################


#####################################################################################
# Add executable
#####################################################################################
include_directories(.)

add_executable(lvr2_example_scan_io
    Main.cpp
)

target_link_libraries(lvr2_example_scan_io
    lvr2_static
)

set_target_properties(lvr2_example_scan_io PROPERTIES
    CXX_STANDARD 17
)

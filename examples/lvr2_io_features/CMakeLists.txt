#####################################################################################
# Package includes
#####################################################################################


#####################################################################################
# Add executable
#####################################################################################
include_directories(.)

add_executable(lvr2_example_io_features
    Main.cpp
)

target_link_libraries(lvr2_example_io_features
    lvr2_static
)

set_target_properties(lvr2_example_io_features PROPERTIES
    CXX_STANDARD 17
)

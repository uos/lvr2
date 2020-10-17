add_executable(lvr2_examples_hdf5features
    Main.cpp
)

target_link_libraries(lvr2_examples_hdf5features
    lvr2_static
)

set_target_properties(lvr2_examples_hdf5features PROPERTIES
    CXX_STANDARD 17
)

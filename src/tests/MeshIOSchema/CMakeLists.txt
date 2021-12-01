#####################################################################################
# Set source files
#####################################################################################
set(MESH_IO_SCHEMA_TEST_SRCS
    Main.cpp
)

include_directories(include)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(MESH_IO_SCHEMA_TEST_DEPS
    lvr2_static
    lvr2las_static
    lvr2rply_static
)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

add_executable(mesh_io_schema_test ${MESH_IO_SCHEMA_TEST_SRCS})
target_link_libraries(mesh_io_schema_test 
    ${MESH_IO_SCHEMA_TEST_DEPS}
)

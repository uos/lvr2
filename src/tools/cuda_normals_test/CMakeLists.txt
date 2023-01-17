#####################################################################################
# Set source files
#####################################################################################

set(RECONSTRUCT_SOURCES
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_RECONSTRUCT_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    ${LVR2_LIB_DEPENDENCIES}
)

if(CUDA_FOUND)
  list(APPEND LVR2_RECONSTRUCT_DEPENDENCIES lvr2cuda_static)
endif(CUDA_FOUND)

#####################################################################################
# Add executable
#####################################################################################
add_executable(cuda_normals_test ${RECONSTRUCT_SOURCES})
target_link_libraries(cuda_normals_test ${LVR2_RECONSTRUCT_DEPENDENCIES})

install(TARGETS cuda_normals_test
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
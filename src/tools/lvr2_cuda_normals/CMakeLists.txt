#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CUDA_NORMAL_SRC
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CUDA_NORMAL_DEPS
    lvr2_static
    lvr2cuda_static
    ${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

add_executable(lvr2_cuda_normals ${LVR2_CUDA_NORMAL_SRC})
target_link_libraries(lvr2_cuda_normals ${LVR2_CUDA_NORMAL_DEPS})

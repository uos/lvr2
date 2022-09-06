#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CUDA_NORMAL_SRC2
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CUDA_NORMAL2_DEPS
    lvr2_static
    #lvr2cuda_static
    ${LVR2_LIB_DEPENDENCIES}
)

cuda_add_library(my_cuda_normals CudaNormals.cu)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

add_executable(lvr2_cuda_normals2 ${LVR2_CUDA_NORMAL_SRC2})
target_link_libraries(lvr2_cuda_normals2 
    ${LVR2_CUDA_NORMAL2_DEPS}
    my_cuda_normals
)

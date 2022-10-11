include_directories(include)


set(CUDA_SEPARABLE_COMPILATION ON)
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

# PURE CUDA
cuda_add_library(kernels_device 
                src/kernels_device.cu
                src/morton_code.cu
                )
target_link_libraries(kernels_device 
    ${CUDA_LIBRARIES}
)

# HEADER HOST - CODE: CUDA calls
cuda_add_library(kernels_host src/kernels_host.cu)
target_link_libraries(kernels_host
    kernels_device
)


# PURE Host
add_executable(lvr2_cuda_normals2 ${LVR2_CUDA_NORMAL_SRC2})
target_link_libraries(lvr2_cuda_normals2 
    ${LVR2_CUDA_NORMAL2_DEPS}
    kernels_host
)
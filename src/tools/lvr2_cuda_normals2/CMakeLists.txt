include_directories(include)


set(CUDA_SEPARABLE_COMPILATION ON)
#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CUDA_NORMAL2_SRC
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
message(STATUS "AAA ${CUDA_INCLUDE_DIRS}")

add_definitions(-DCUDA_INCLUDE_DIRS="${CUDA_INCLUDE_DIRS}")
cuda_add_library(kernels_device 
                src/kernels_device.cu
                src/morton_code.cu
                src/LBVHIndex.cu
                src/lbvh_kernels.cu
                src/normals_kernel.cu
                #src/LBVHTree.cu
                #src/query_knn_kernels.cu
                #src/query_knn.cu
                #src/query.cu
                )
target_link_libraries(kernels_device 
    ${CUDA_LIBRARIES}
    # TODO Find nvrtc library
    /usr/local/cuda/lib64/libnvrtc.so
    ${CUDA_CUDA_LIBRARY}
)

# HEADER HOST - CODE: CUDA calls
cuda_add_library(kernels_host 
                src/kernels_host.cu
                )
target_link_libraries(kernels_host
    kernels_device
)


# PURE Host
add_executable(lvr2_cuda_normals2 ${LVR2_CUDA_NORMAL2_SRC})
target_link_libraries(lvr2_cuda_normals2 
    ${LVR2_CUDA_NORMAL2_DEPS}
    kernels_host
)
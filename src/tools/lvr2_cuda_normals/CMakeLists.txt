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
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    lvr2_static
    lvr2las_static
    lvr2rply_static
    lvr2slam6d_static
    lvr2cuda_static
)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

add_executable(lvr2_cuda_normals ${LVR2_CUDA_NORMAL_SRC})

target_link_libraries(lvr2_cuda_normals ${LVR2_CUDA_NORMAL_DEPS} ${CUDA_LIBRARIES})

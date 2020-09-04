#####################################################################################
# RAYCASTER EXAMPLE
#####################################################################################

set(RAYCASTER_DEPS lvr2_static)

# if(OPENCL_FOUND)
#     list(APPEND RAYCASTER_DEPS ${OpenCL_LIBRARIES})
# endif(OPENCL_FOUND)

# if(embree_FOUND)
#     list(APPEND RAYCASTER_DEPS ${EMBREE_LIBRARY})
# endif(embree_FOUND)

# Add executable
add_executable(lvr2_examples_raycasting
    Main.cpp
)

# link
target_link_libraries(lvr2_examples_raycasting
    ${RAYCASTER_DEPS}
)

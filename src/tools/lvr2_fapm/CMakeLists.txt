#####################################################################################
# RAYCASTER EXAMPLE
#####################################################################################

set(LVR2_FAMP_DEPS lvr2_static)

if(OPENCL_FOUND)
    list(APPEND LVR2_FAMP_DEPS ${OpenCL_LIBRARIES})
endif(OPENCL_FOUND)

if(embree_FOUND)
    list(APPEND LVR2_FAMP_DEPS ${EMBREE_LIBRARY})
endif(embree_FOUND)

# Add executable
add_executable(lvr2_fapm
    Main.cpp
)

# link
target_link_libraries(lvr2_fapm
    ${LVR2_FAMP_DEPS}
)

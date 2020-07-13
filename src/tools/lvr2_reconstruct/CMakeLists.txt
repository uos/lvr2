#####################################################################################
# Set source files
#####################################################################################

set(RECONSTRUCT_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_RECONSTRUCT_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    lvr2slam6d_static
    ${LVR2_LIB_DEPENDENCIES}
)

if(CUDA_FOUND)
  list(APPEND LVR2_RECONSTRUCT_DEPENDENCIES lvr2cuda_static)
endif(CUDA_FOUND)

if( ${NABO_FOUND} )
  list(APPEND LVR2_RECONSTRUCT_DEPENDENCIES ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_reconstruct ${RECONSTRUCT_SOURCES})
target_link_libraries(lvr2_reconstruct ${LVR2_RECONSTRUCT_DEPENDENCIES})

install(TARGETS lvr2_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

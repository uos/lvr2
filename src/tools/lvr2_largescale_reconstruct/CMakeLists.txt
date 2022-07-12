#####################################################################################
# Set source files
#####################################################################################

set(LS_RECONSTRUCT_SOURCES
    Options.cpp
    Main.cpp)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_LS_RECONSTRUCT_DEPENDENCIES
    lvr2_static
    ${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_largescale_reconstruct ${LS_RECONSTRUCT_SOURCES} )

target_link_libraries(lvr2_largescale_reconstruct ${LVR2_LS_RECONSTRUCT_DEPENDENCIES} )


if(CUDA_FOUND)
        target_link_libraries(lvr2_largescale_reconstruct lvr2cuda TBB::tbb)
        target_compile_definitions(lvr2_largescale_reconstruct PRIVATE CUDA_FOUND=1)
elseif(OPENCL_FOUND)
        target_compile_definitions(lvr2_largescale_reconstruct PRIVATE OPENCL_FOUND=1)
endif()

install(TARGETS lvr2_largescale_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

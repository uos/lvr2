#####################################################################################
# Set source files
#####################################################################################

set(LVR2_CHUNKING_SERVER_SOURCES
    Main.cpp
    Options.cpp
    #../../../include/lvr2/reconstruction/LargeScaleReconstruction.tcc
    #    ../../../include/lvr2/reconstruction/LargeScaleReconstruction.hpp
    #           ../../../src/liblvr2/algorithm/ChunkingPipeline.cpp
        )

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_CHUNKING_SERVER_DEPENDENCIES
    lvr2
    lvr2las_static
    lvr2rply_static
    ${Boost_LIBRARIES}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

if(CUDA_FOUND)
    set(LVR2_CHUNKING_SERVER_DEPENDENCIES
        ${LVR2_CHUNKING_SERVER_DEPENDENCIES}
        lvr2cuda_static
    )
else()
    set(LVR2_CHUNKING_SERVER_DEPENDENCIES
        ${LVR2_CHUNKING_SERVER_DEPENDENCIES}
        lvr2_static
    )
endif()

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_chunking_server ${LVR2_CHUNKING_SERVER_SOURCES})
target_link_libraries(lvr2_chunking_server ${LVR2_CHUNKING_SERVER_DEPENDENCIES})

if(CUDA_FOUND)
        target_link_libraries(lvr2_chunking_server lvr2cuda)
        target_compile_definitions(lvr2_chunking_server PRIVATE CUDA_FOUND=1)
elseif(OPENCL_FOUND)
        target_compile_definitions(lvr2_chunking_server PRIVATE OPENCL_FOUND=1)
endif()

install(TARGETS lvr2_chunking_server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

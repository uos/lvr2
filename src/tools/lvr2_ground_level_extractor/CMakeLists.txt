#####################################################################################
# Set source files
#####################################################################################

set(GROUND_LEVEL_EXTRACTOR_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    ${LVR2_LIB_DEPENDENCIES}
)

if( ${NABO_FOUND} )
  set(LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES ${LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES  ${LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_ground_level_extractor ${GROUND_LEVEL_EXTRACTOR_SOURCES})
target_link_libraries(lvr2_ground_level_extractor ${LVR2_GROUND_LEVEL_EXTRACTOR_DEPENDENCIES})
target_precompile_headers(lvr2_ground_level_extractor PUBLIC  ${LVR2_PCHS})
find_package(HDF5 QUIET REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
target_link_libraries(lvr2_ground_level_extractor ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

if(CUDA_FOUND)
        target_link_libraries(lvr2_ground_level_extractor lvr2cuda)
        target_compile_definitions(lvr2_ground_level_extractor PRIVATE CUDA_FOUND=1)
endif()
if(OPENCL_FOUND)
        target_compile_definitions(lvr2_ground_level_extractor PRIVATE OPENCL_FOUND=1)
endif()


install(TARGETS lvr2_ground_level_extractor
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

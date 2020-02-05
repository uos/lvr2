#####################################################################################
# Set source files
#####################################################################################

set(LVR_REDUCE_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_REDUCE_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    lvr2slam6d_static
    ${LVR2_LIB_DEPENDENCIES}
)

if( ${NABO_FOUND} )
  set(LVR_REDUCE_DEPENDENCIES ${LVR_REDUCE_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR_REDUCE_DEPENDENCIES  ${LVR_REDUCE_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_mesh_reduce ${LVR_REDUCE_SOURCES})
#set_target_properties(lvr2_mesh_reduce PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr2_mesh_reduce ${LVR_REDUCE_DEPENDENCIES})

find_package(HDF5 QUIET REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
target_link_libraries(lvr2_mesh_reduce ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

if(CUDA_FOUND)
        target_link_libraries(lvr2_mesh_reduce lvr2cuda)
        target_compile_definitions(lvr2_mesh_reduce PRIVATE CUDA_FOUND=1)
endif()
if(OPENCL_FOUND)
        target_compile_definitions(lvr2_mesh_reduce PRIVATE OPENCL_FOUND=1)
endif()


install(TARGETS lvr2_mesh_reduce
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

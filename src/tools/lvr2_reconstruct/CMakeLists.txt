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

set(LVR_RECONSTRUCT_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    lvr2slam6d_static
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

if( ${NABO_FOUND} )
  set(LVR_RECONSTRUCT_DEPENDENCIES ${LVR_RECONSTRUCT_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR_RECONSTRUCT_DEPENDENCIES  ${LVR_RECONSTRUCT_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_reconstruct ${RECONSTRUCT_SOURCES})
#set_target_properties(lvr2_reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr2_reconstruct ${LVR_RECONSTRUCT_DEPENDENCIES})

find_package(HDF5 QUIET REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
target_link_libraries(lvr2_reconstruct ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

if(CUDA_FOUND)
        target_link_libraries(lvr2_reconstruct lvr2cuda)
        target_compile_definitions(lvr2_reconstruct PRIVATE CUDA_FOUND=1)
endif()
if(OPENCL_FOUND)
        target_compile_definitions(lvr2_reconstruct PRIVATE OPENCL_FOUND=1)
endif()


install(TARGETS lvr2_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

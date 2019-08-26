

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_SCANPROJECTTOHDF_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    lvr2slam6d_static
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR_SCANPROJECTTOHDF_DEPENDENCIES  ${LVR_SCANPROJECTTOHDF_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_scanproject2hdf5
  Main.cpp
)
#set_target_properties(lvr2_reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr2_scanproject2hdf5
  ${LVR_SCANPROJECTTOHDF_DEPENDENCIES}
)

find_package(HDF5 QUIET REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
target_link_libraries(lvr2_scanproject2hdf5
  ${HDF5_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
)

install(TARGETS lvr2_scanproject2hdf5
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

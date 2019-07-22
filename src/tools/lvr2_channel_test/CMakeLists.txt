#####################################################################################
# Set source files
#####################################################################################

include_directories(
  include
)

set(LVR2_IOPATCH_SOURCES
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_IOPATCH_DEPENDENCIES
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
  set(LVR2_IOPATCH_DEPENDENCIES ${LVR2_IOPATCH_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR2_IOPATCH_DEPENDENCIES  ${LVR2_IOPATCH_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)

find_package(HDF5 QUIET REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})

#####################################################################################
# Add executable
#####################################################################################

# add_executable(lvr2_io_patch_test
#     Main.cpp
#     ${LVR2_IOPATCH_SOURCES}
# )
# #set_target_properties(lvr2_reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# target_link_libraries(lvr2_io_patch_test ${LVR2_IOPATCH_DEPENDENCIES})
# target_link_libraries(lvr2_io_patch_test ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

add_executable(lvr2_channel_test
    ChannelTest.cpp
    ${LVR2_IOPATCH_SOURCES}
)

target_link_libraries(lvr2_channel_test ${LVR2_IOPATCH_DEPENDENCIES})
target_link_libraries(lvr2_channel_test ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})



# install(TARGETS lvr2_io_patch_test
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



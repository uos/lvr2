#####################################################################################
# Set source files
#####################################################################################

set(
        LVR_CONVERTER_SOURCES
        test.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(
        LVR_CONVERTER_DEPENDENCIES
        lvr2_static
        lvrlas_static
        lvrrply_static
        lvrslam6d_static
        ${TIFF_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${OpenCV_LIBS}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5-tiff-converter ${LVR_CONVERTER_SOURCES} )
target_link_libraries( lvr2_hdf5-tiff-converter ${LVR_CONVERTER_DEPENDENCIES} )

install(TARGETS lvr2_hdf5-tiff-converter RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

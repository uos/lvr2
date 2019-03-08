#####################################################################################
# set source files
#####################################################################################

set(
        LVR_CONVERTER_SOURCES
        Main.cpp
        GeoTIFFIO.cpp GeoTIFFIO.hpp
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
        ${GDAL_LIBRARY}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${OpenCV_LIBS}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_classification-converter ${LVR_CONVERTER_SOURCES} )
target_link_libraries( lvr2_classification-converter ${LVR_CONVERTER_DEPENDENCIES} )

install(TARGETS lvr2_classification-converter RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

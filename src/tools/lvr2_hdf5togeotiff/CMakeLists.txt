#####################################################################################
# set source files
#####################################################################################

set(
        LVR_CONVERTER_SOURCES
        Main.cpp
        ../../liblvr2/io/GeoTIFFIO.cpp ../../../include/lvr2/io/GeoTIFFIO.hpp
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

add_executable( lvr2_hdf5togeotiff ${LVR_CONVERTER_SOURCES} )
target_link_libraries( lvr2_hdf5togeotiff ${LVR_CONVERTER_DEPENDENCIES} )

install(TARGETS lvr2_hdf5togeotiff RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

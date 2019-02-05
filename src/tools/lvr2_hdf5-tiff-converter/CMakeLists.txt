#####################################################################################
# cmake module for GeoTIFF
# source: https://github.com/OSGeo/libgeotiff/blob/master/libgeotiff/cmake/FindGeoTIFF.cmake
#####################################################################################

#[[IF(UNIX)
    FIND_PATH(GEOTIFF_INCLUDE_DIR geotiff.h PATH_PREFIXES geotiff)

    FIND_LIBRARY(GEOTIFF_LIBRARY NAMES ${GEOTIFF_NAMES})

ELSE()
    MESSAGE("FindGeoTIFF.cmake: Not a unix operating system.")
ENDIF()

IF(GEOTIFF_FOUND)
    SET(GEOTIFF_LIBRARIES ${GEOTIFF_LIBRARY})
ENDIF()

# Handle the QUIETLY and REQUIRED arguments and set SPATIALINDEX_FOUND to TRUE
# if all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GEOTIFF DEFAULT_MSG GEOTIFF_LIBRARY GEOTIFF_INCLUDE_DIR)]]

#####################################################################################
# set source files
#####################################################################################

set(
        LVR_CONVERTER_SOURCES
        Main.cpp
        TIFFIO.cpp TIFFIO.hpp)

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

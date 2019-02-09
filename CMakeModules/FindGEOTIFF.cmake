#####################################################################################
# cmake module for GeoTIFF
# source: https://github.com/OSGeo/libgeotiff/blob/master/libgeotiff/cmake/FindGeoTIFF.cmake
# NOTE: The assignment GEOTIFF_LIBRARIES ${GEOTIFF_LIBRARY} does not work.
#       Need to use ${GEOTIFF_LIBRARY} in CMakeLists instead.
#####################################################################################

set(GEOTIFF_NAMES geotiff)

find_path(GEOTIFF_INCLUDE_DIR geotiff.h PATH_PREFIXES geotiff)

find_library(GEOTIFF_LIBRARY NAMES ${GEOTIFF_NAMES})

# Handle the QUIETLY and REQUIRED arguments and set SPATIALINDEX_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GEOTIFF DEFAULT_MSG GEOTIFF_LIBRARY GEOTIFF_INCLUDE_DIR)

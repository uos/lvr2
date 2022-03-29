if(NOT CESIUM_NATIVE_FOUND)
    message(WARNING "lvr2_3dtiles requires cesium-native in ext directory to be built. Skipping lvr2_3dtiles")
    return()
endif()

#####################################################################################
# Set source files
#####################################################################################

set(TILES_SOURCES
    Main.cpp
    B3dmWriter.cpp
    Segmenter.cpp
    SegmentTree.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_3DTILES_DEPENDENCIES
    lvr2_static
    Cesium3DTiles
    Cesium3DTilesWriter
    CesiumGltf
    CesiumGltfWriter
    ${LVR2_LIB_DEPENDENCIES}
)

get_target_property(CESIUM_INCLUDES Cesium3DTiles INCLUDE_DIRECTORIES)
include_directories(${CESIUM_INCLUDES})
get_target_property(CESIUM_INCLUDES Cesium3DTilesWriter INCLUDE_DIRECTORIES)
include_directories(${CESIUM_INCLUDES})
get_target_property(CESIUM_INCLUDES CesiumGltf INCLUDE_DIRECTORIES)
include_directories(${CESIUM_INCLUDES})
get_target_property(CESIUM_INCLUDES CesiumGltfWriter INCLUDE_DIRECTORIES)
include_directories(${CESIUM_INCLUDES})

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_3dtiles ${TILES_SOURCES})
target_link_libraries(lvr2_3dtiles ${LVR2_3DTILES_DEPENDENCIES})

install(TARGETS lvr2_3dtiles
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

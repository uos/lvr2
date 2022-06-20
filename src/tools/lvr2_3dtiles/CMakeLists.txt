if(NOT WITH_3DTILES)
    add_custom_target(lvr2_3dtiles
                      COMMAND echo "ERROR: building lvr2_3dtiles requires  -DWITH_3DTILES=ON  cmake flag.")
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
    ${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_3dtiles ${TILES_SOURCES})
target_link_libraries(lvr2_3dtiles ${LVR2_3DTILES_DEPENDENCIES})

install(TARGETS lvr2_3dtiles
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

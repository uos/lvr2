#####################################################################################
# set source files
#####################################################################################

set(
        LVR_CONVERTER_SOURCES
        Main.cpp
        Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(
        LVR_CONVERTER_DEPENDENCIES
        lvr2_static
        ${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5togeotiff ${LVR_CONVERTER_SOURCES} )
target_link_libraries( lvr2_hdf5togeotiff ${LVR_CONVERTER_DEPENDENCIES} )

install(TARGETS lvr2_hdf5togeotiff RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#####################################################################################
# Set source files
#####################################################################################

set(LVR2_HDF5_INSPECT_SOURCES
	Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_HDF5_INSPECT_DEPS
	lvr2_static
	lvr2las_static
	lvr2rply_static
	${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5_inspect ${LVR2_HDF5_INSPECT_SOURCES} )
target_link_libraries( lvr2_hdf5_inspect ${LVR2_HDF5_INSPECT_DEPS} )

install(TARGETS lvr2_hdf5_inspect
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

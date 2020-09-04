#####################################################################################
# Set source files
#####################################################################################

set(DMC_RECONSTRUCTION_SOURCES
        Main.cpp
        Options.cpp
		)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(DMC_DEPENDENCIES
		lvr2_static
		lvr2las_static
		lvr2rply_static
		${LVR2_LIB_DEPENDENCIES}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_dmc_reconstruction ${DMC_RECONSTRUCTION_SOURCES})
target_link_libraries(lvr2_dmc_reconstruction ${DMC_DEPENDENCIES} boost_date_time ${Boost_LOG_LIBRARY_RELEASE} lz4)

install(TARGETS lvr2_dmc_reconstruction
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#####################################################################################
# Set source files
#####################################################################################

set(FASTSENSE_RECONSTRUCTION_SOURCES
        Main.cpp
		)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(FASTSENSE_DEPENDENCIES
		lvr2_static
		lvr2las_static
		lvr2rply_static
		${LVR2_LIB_DEPENDENCIES}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_fastsense_reconstruction ${FASTSENSE_RECONSTRUCTION_SOURCES})
target_link_libraries(lvr2_fastsense_reconstruction ${FASTSENSE_DEPENDENCIES} boost_date_time ${Boost_LOG_LIBRARY_RELEASE} lz4)

install(TARGETS lvr2_fastsense_reconstruction
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

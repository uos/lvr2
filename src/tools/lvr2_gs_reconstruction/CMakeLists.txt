#####################################################################################
# Set source files
#####################################################################################

set(GS_RECONSTRUCTION_SOURCES
        Main.cpp
		OptionsGSS.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(GS_DEPENDENCIES
	lvr2_static
	lvrlas_static
	lvrrply_static
  ${PCL_LIBRARIES}
  ${CGAL_LIBRARY}
  ${EIGEN_LIBRARY}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_gs_reconstruction ${GS_RECONSTRUCTION_SOURCES})
target_link_libraries(lvr2_gs_reconstruction ${GS_DEPENDENCIES} boost_date_time ${Boost_LOG_LIBRARY_RELEASE} lz4)

install(TARGETS lvr2_gs_reconstruction
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

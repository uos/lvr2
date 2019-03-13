#####################################################################################
# Set source files
#####################################################################################

set(GSS_RECONSTRUCT_SOURCES
        Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(GSS_DEPENDENCIES
	lvr2_static
	lvrlas_static
	lvrrply_static
  ${PCL_LIBRARIES}
  ${EIGEN_LIBRARY}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_gss_reconstruct ${GSS_RECONSTRUCT_SOURCES})
target_link_libraries(lvr2_gss_reconstruct ${GSS_DEPENDENCIES} boost_date_time ${Boost_LOG_LIBRARY_RELEASE} lz4)

install(TARGETS lvr2_gss_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

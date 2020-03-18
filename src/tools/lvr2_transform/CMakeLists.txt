#####################################################################################
# Set source files
#####################################################################################

set(LVR2_TRANSFORM_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_TRANSFORM_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
	)

if( ${NABO_FOUND} )
  set(LVR2_TRANSFORM_DEPENDENCIES ${LVR2_TRANSFORM_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_transform_model ${LVR2_TRANSFORM_SOURCES})
target_link_libraries(lvr2_transform_model ${LVR2_TRANSFORM_DEPENDENCIES})

install(TARGETS lvr2_transform_model
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

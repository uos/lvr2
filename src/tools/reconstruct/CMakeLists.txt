#####################################################################################
# Set source files
#####################################################################################

set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_RECONSTRUCT_DEPENDENCIES 
	lvr_static
	${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} 
	${Boost_THREAD_LIBRARY} 
	${Boost_FILESYSTEM_LIBRARY}
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	)

if( ${NABO_FOUND} )
set(LVR_RECONSTRUCT_DEPENDENCIES ${LVR_RECONSTRUCT_DEPENDENCIES} ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_reconstruct ${RECONSTRUCT_SOURCES})
set_target_properties(lvr_reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_reconstruct ${LVR_RECONSTRUCT_DEPENDENCIES})


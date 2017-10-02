
#####################################################################################
# Set source files
#####################################################################################

set(LVR_TIOGEN_SOURCES
   tiogen.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_TIOGEN_DEPENDENCIES
    lvr_static
    lvrlas_static
    lvrrply_static
    lvrslam6d_static
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    )

if( ${NABO_FOUND} )
  set(LVR_TIOGEN_DEPENDENCIES ${LVR_TIOGEN_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr_tiogen ${LVR_TIOGEN_SOURCES} )
target_link_libraries( lvr_tiogen ${LVR_TIOGEN_DEPENDENCIES} )

install(TARGETS lvr_tiogen
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
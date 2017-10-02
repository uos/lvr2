#####################################################################################
# Set source files
#####################################################################################

set(KABOOM_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_KABOOM_DEPENDENCIES
    lvr_static
    lvrlas_static
    lvrrply_static
    lvrslam6d_static
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    )

if(PCL_FOUND)
    set(LVR_KABOOM_DEPENDENCIES ${LVR_LEICA_DEPENDENCIES} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_kaboom ${KABOOM_SOURCES})
target_link_libraries(lvr_kaboom ${LVR_KABOOM_DEPENDENCIES})

install(TARGETS lvr_kaboom
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
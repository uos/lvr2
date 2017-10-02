#####################################################################################
# Set source files
#####################################################################################

set(LVR_IMG_NORMAL_SRC
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_IMG_NORMAL_DEPS
    lvr_static
    lvrlas_static
    lvrrply_static
    lvrslam6d_static
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    )

if(PCL_FOUND)
  set(LVR_IMG_NORMAL_DEPS  ${LVR_IMG_NORMAL_DEPS} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

if( ${NABO_FOUND} )
 set(LVR_IMG_NORMAL_DEPS  ${LVR_IMG_NORMAL_DEPS} ${NABO_LIBRARY} )
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_image_normals ${LVR_IMG_NORMAL_SRC})
target_link_libraries(lvr_image_normals ${LVR_IMG_NORMAL_DEPS})

install(TARGETS lvr_image_normals
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
#####################################################################################
# Set source files
#####################################################################################

set(LVR_IMG_NORMAL_SRC
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/ModelFactory.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/AttributeManager.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/BaseBuffer.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PointBuffer2.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/MeshBuffer2.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/Progress.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/BaseIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/Timestamp.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PLYIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/ObjIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PPMIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/AsciiIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/texture/Texture.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/display/TextureFactory.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/display/GlTexture.cpp
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

add_executable(lvr2_image_normals ${LVR_IMG_NORMAL_SRC})
target_link_libraries(lvr2_image_normals ${LVR_IMG_NORMAL_DEPS})

install(TARGETS lvr2_image_normals
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

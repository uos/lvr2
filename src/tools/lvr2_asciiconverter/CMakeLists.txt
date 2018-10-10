#####################################################################################
# Set source files
#####################################################################################

set(ASCII_SOURCES
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/ModelFactory.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/AttributeManager.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/BaseBuffer.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PointBuffer2.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/MeshBuffer2.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/Progress.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/BaseIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/Timestamp.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/AsciiIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PLYIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/util/Util.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/ObjIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PPMIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/texture/Texture.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/texture/TextureFactory.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/display/GlTexture.cpp
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_ASCIICONVERT_DEPENDENCIES
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_ascii_convert ${ASCII_SOURCES})
target_link_libraries(lvr2_ascii_convert ${LVR_ASCIICONVERT_DEPENDENCIES})

install(TARGETS lvr2_ascii_convert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#####################################################################################
# Set source files
#####################################################################################

set(PLY_MERGE_SRC
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
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/ObjIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/PPMIO.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/texture/Texture.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/display/TextureFactory.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/display/GlTexture.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/util/Util.cpp
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_PLY_MERGER_DEPENDENCIES
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

add_executable(lvr2_ply_merger ${PLY_MERGE_SRC})
target_link_libraries(lvr2_ply_merger ${LVR_PLY_MERGER_DEPENDENCIES})

install(TARGETS lvr2_ply_merger
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#####################################################################################
# Set source files
#####################################################################################

find_package(OpenCL REQUIRED)
include_directories(OpenCL_INCLUDE_DIRS)
message(STATUS "Loaded OpenCL version ${OpenCL_VERSION_STRING}")
message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")

set(LVR_CL_NORMAL_SRC
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
    ${PROJECT_SOURCE_DIR}/include/lvr2/reconstruction/LBKdTree.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/io/IOUtils.cpp
    ${PROJECT_SOURCE_DIR}/include/lvr2/util/Util.cpp
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_CL_NORMAL_DEPS
    lvr
    ${OpenCL_LIBRARIES}
)



#####################################################################################
# Add executable
#####################################################################################
add_executable(lvr2_cl_normals ${LVR_CL_NORMAL_SRC})

target_link_libraries(lvr2_cl_normals ${LVR_CL_NORMAL_DEPS})

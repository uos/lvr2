#####################################################################################
# Set source files
#####################################################################################

set(LVR_CUDA_NORMAL_SRC
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

set(LVR_CUDA_NORMAL_DEPS
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    lvr_static
    lvrlas_static
    lvrrply_static
    lvrslam6d_static
    lvrcuda_static
)

#####################################################################################
# Add executable
#####################################################################################

###### ADD YOUR CODE HERE #######

# Fix broken VTK flags
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

cuda_add_library(cuda_normals STATIC ${PROJECT_SOURCE_DIR}/include/lvr2/reconstruction/cuda/CudaSurface.cu)

add_executable(lvr2_cuda_normals ${LVR_CUDA_NORMAL_SRC})

target_link_libraries(lvr2_cuda_normals ${LVR_CUDA_NORMAL_DEPS} ${CUDA_LIBRARIES} cuda_normals)

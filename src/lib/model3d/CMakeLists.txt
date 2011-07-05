set(LIB_3D_SOURCES 
Arrow.cpp
BaseVertex.cpp
BoundingBox.cpp
ColorVertex.cpp
CoordinateAxes.cpp
GroundPlane.cpp
HalfEdge.cpp
HalfEdgeFace.cpp
HalfEdgeMesh.cpp
HalfEdgePolygon.cpp
HalfEdgeVertex.cpp
LinkedTriangle.cpp
LinkedTriangleMesh.cpp
LinkedVertex.cpp
Matrix4.cpp
NormalCloud.cpp
Normal.cpp
MultiPointCloud.cpp
OBJReader.cpp
PLYElement.cpp
PLYElements.cpp
PLYProperty.cpp
PLYWriter.cpp
PointCloud.cpp
PointCloud.cpp
Polygon.cpp
PolygonMesh.cpp
ProgressiveMesh.cpp
Quaternion.cpp 
Renderable.cpp   
StaticMesh.cpp
Triangle.cpp
TriangleMesh.cpp
Tube.cpp)

add_library(model3d SHARED ${LIB_3D_SOURCES})
target_link_libraries(model3d ${OPENGL_LIBRARY})
target_link_libraries(model3d ${Boost_FILESYSTEM_LIBRARY})
set_target_properties(model3d PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

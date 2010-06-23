de###############################################################################
#
#	MAKEFILE FOR THE 3D MODELLING TOOLKIT 
#   -------------------------------------
#
#   Author: Thomas Wiemann
#   Date  : 03/23/2009
#
###############################################################################

include ../../Makefile.options

OBJ = ../../obj/
BIN = ../../bin/
LIB = ../../lib/

LIB3DTARGETS = $(OBJ)BaseVertex.o $(OBJ)ColorVertex.o \
               $(OBJ)CoordinateAxes.o $(OBJ)GroundPlane.o $(OBJ)Matrix4.o \
               $(OBJ)Normal.o $(OBJ)NormalCloud.o $(OBJ)PointCloud.o \
               $(OBJ)Quaternion.o $(OBJ)Renderable.o $(OBJ)StaticMesh.o \
               $(OBJ)TriangleMesh.o $(OBJ)Tube.o $(OBJ)BoundingBox.o \
               $(OBJ)HalfEdgeMesh.o $(OBJ)HalfEdgePolygon.o $(OBJ)HalfEdge.o \
               $(OBJ)HalfEdgeVertex.o $(OBJ)HalfEdgeFace.o $(OBJ)Triangle.o \
               $(OBJ)LinkedVertex.o $(OBJ)LinkedTriangle.o \
               $(OBJ)LinkedTriangleMesh.o $(OBJ)PolygonMesh.o $(OBJ)Polygon.o \
               $(OBJ)PLYElements.o $(OBJ)PLYWriter.o $(OBJ)PLYProperty.o \
               $(OBJ)PLYElement.o

all: libmodel.a 


clean:
	@echo "Cleaning Modelling Toolkit's object files and library..."
	rm -rf $(LIB3DTARGETS)

############################ 3D MODELLING LIBRARY #############################

libmodel.a: $(LIB3DTARGETS)
	@echo -e "\nLinking 3D Modelling Toolkit Library...\n"
	@$(AR) -cr $(LIB)libmodel.a $(LIB3DTARGETS)
	@ranlib  $(LIB)libmodel.a

############################### OBJECT FILES ##################################

$(OBJ)BaseVertex.o: BaseVertex.*
	@echo "Compiling Base Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)BaseVertex.o BaseVertex.cpp

$(OBJ)BoundingBox.o: BoundingBox.*
	@echo "Compiling Bounding Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)BoundingBox.o BoundingBox.cpp

$(OBJ)ColorVertex.o: ColorVertex.*
	@echo "Compiling Color Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ColorVertex.o ColorVertex.cpp

$(OBJ)Normal.o: Normal.*
	@echo "Compiling Normal..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Normal.o Normal.cpp

$(OBJ)Matrix4.o: Matrix4.*
	@echo "Compiling 4x4 Matrix..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Matrix4.o Matrix4.cpp

$(OBJ)Renderable.o: Renderable.*
	@echo "Compiling Renderable..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Renderable.o Renderable.cpp

$(OBJ)Tube.o: Tube.*
	@echo "Compiling Tube..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Tube.o Tube.cpp	

$(OBJ)Triangle.o: Triangle.*
	@echo "Compiling Triangle..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Triangle.o Triangle.cpp

$(OBJ)GroundPlane.o: GroundPlane.*
	@echo "Compiling Ground Plane..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)GroundPlane.o GroundPlane.cpp

$(OBJ)CoordinateAxes.o: CoordinateAxes.*
	@echo "Compiling Coordinate Axes..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)CoordinateAxes.o CoordinateAxes.cpp

$(OBJ)Quaternion.o: Quaternion.*
	@echo "Compiling Quaternion..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Quaternion.o Quaternion.cpp

$(OBJ)PointCloud.o: PointCloud.*
	@echo "Compiling Point Cloud..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PointCloud.o PointCloud.cpp

$(OBJ)NormalCloud.o: NormalCloud.*
	@echo "Compiling Normal Point Cloud..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)NormalCloud.o NormalCloud.cpp

$(OBJ)StaticMesh.o: StaticMesh.*
	@echo "Compiling Static Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)StaticMesh.o StaticMesh.cpp

$(OBJ)TriangleMesh.o: TriangleMesh.*
	@echo "Compiling Triangle Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TriangleMesh.o TriangleMesh.cpp

$(OBJ)HalfEdgeMesh.o: HalfEdgeMesh.*
	@echo "Compiling Half Edge Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HalfEdgeMesh.o HalfEdgeMesh.cpp

$(OBJ)HalfEdgePolygon.o: HalfEdgePolygon.*
	@echo "Compiling Half Edge Polygon..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HalfEdgePolygon.o HalfEdgePolygon.cpp

$(OBJ)HalfEdge.o: HalfEdge.*
	@echo "Compiling Half Edge..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HalfEdge.o HalfEdge.cpp

$(OBJ)HalfEdgeVertex.o: HalfEdgeVertex.*
	@echo "Compiling Half Edge Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HalfEdgeVertex.o HalfEdgeVertex.cpp

$(OBJ)HalfEdgeFace.o: HalfEdgeFace.*
	@echo "Compiling Half Edge Face..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HalfEdgeFace.o HalfEdgeFace.cpp

$(OBJ)LinkedVertex.o: LinkedVertex.*
	@echo "Compiling Linked Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)LinkedVertex.o LinkedVertex.cpp

$(OBJ)LinkedTriangleMesh.o: LinkedTriangleMesh.* LinkedTriangle.* LinkedVertex.*
	@echo "Compiling Linked Triangle Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)LinkedTriangleMesh.o LinkedTriangleMesh.cpp

$(OBJ)LinkedTriangle.o: LinkedTriangle.*
	@echo "Compiling Linked Triangle..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)LinkedTriangle.o LinkedTriangle.cpp

$(OBJ)ProgressiveMesh.o: ProgressiveMesh.*
	@echo "Compiling Progressive Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ProgressiveMesh.o ProgressiveMesh.cpp

$(OBJ)PolygonMesh.o: PolygonMesh.*
	@echo "Compiling Polygon Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PolygonMesh.o PolygonMesh.cpp

$(OBJ)Polygon.o: Polygon.*
	@echo "Compiling Polygon..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Polygon.o Polygon.cpp
	
$(OBJ)PLYWriter.o: PLYWriter.*
	@echo "Compiling PLYWriter..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PLYWriter.o $(SRC)PLYWriter.cpp

$(OBJ)PLYElements.o: PLYElements.*
	@echo "Compiling PLYElements..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PLYElements.o $(SRC)PLYElements.cpp
	
$(OBJ)PLYProperty.o: $(SRC)PLYProperty.*
	@echo "Compiling PLY Property..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PLYProperty.o $(SRC)PLYProperty.cpp
	
$(OBJ)PLYElement.o: $(SRC)PLYElement.*
	@echo "Compiling PLY Element..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PLYElement.o $(SRC)PLYElement.cpp
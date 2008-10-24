###############################################################################
#
# SURFACE RECONSTRUCTION TOOLKIT
# -----------------------------------
#
# created on: 01.03.2008
# author:     Thomas Wiemann
#
###############################################################################


CPP = g++
CFLAGS = -Wall -Wno-deprecated -Wno-write-strings -fopenmp 
AR = ar
MOC = moc

BIN = bin/
OBJ = obj/

OSXFLAGS    = #-DMAC_OSX # TO DO: AUTO SELECTION

ifndef $(MACVERSION)
GLLIBS     = -lGL -lGLU
QTLIBS     = -lQtGui -lQtOpenGL
QTINCLUDES = -I/usr/include/QtNetwork -I/usr/include/QtGui -I/usr/include/QtXml \
             -I/usr/include/QtSql -I/usr/include/QtUiTools \
             -I/usr/include/QtOpenGL
else
GLLIBS     = -framework OpenGL
QTLIBS     = -framework QTKit
QTINCLUDES = -I/usr/local/Trolltech/Qt-4.4.3/include/ \
             -I/usr/local/Trolltech/Qt-4.4.3/include/QtGui/ \
             -I/usr/local/Trolltech/Qt-4.4.3/include/QtOpenGL/ \
             -F/usr/local/Trolltech/Qt-4.4.3/lib/
QTLIBS     = -framework QtGui -framework QtOpenGL -framework QtCore
endif

MCSRC      = src/libmc/
MESHSRC    = src/mesh/
ANNSRC     = src/ann/
IOSRC      = src/io/
GSLSRC     = src/gsl/
NMSRC      = src/newmat/
TESTSRC    = src/test/
LIB3DSRC   = src/lib3d/
VIEWSRC    = src/viewer/

ifeq ($(OSXFLAGS), -DMAC_OSX)
GLLIBS     = -framework OpenGL -framework GLUT
else
GLLIBS     = -lGL -lGLU -lglut -lgltt -lttf -lgle
endif

CFLAGS     += -I$(ANNSRC) -I$(GSLSRC)
CFLAGS     += $(OSXFLAGS) $(QTINCLUDES) 

LIB3DTARGETS = $(OBJ)BaseVertex.o $(OBJ)ColorVertex.o \
               $(OBJ)CoordinateAxes.o $(OBJ)GroundPlane.o $(OBJ)Matrix4.o \
               $(OBJ)Normal.o $(OBJ)NormalCloud.o $(OBJ)PointCloud.o \
               $(OBJ)Quaternion.o $(OBJ)Renderable.o $(OBJ)StaticMesh.o \
               $(OBJ)TriangleMesh.o $(OBJ)Tube.o $(OBJ)BoundingBox.o

ANNTARGETS   = $(OBJ)ANN.o $(OBJ)brute.o $(OBJ)kd_tree.o $(OBJ)kd_util.o \
               $(OBJ)kd_split.o $(OBJ)kd_search.o $(OBJ)kd_pr_search.o \
               $(OBJ)kd_fix_rad_search.o $(OBJ)kd_dump.o $(OBJ)bd_tree.o \
               $(OBJ)bd_search.o $(OBJ)bd_pr_search.o \
               $(OBJ)bd_fix_rad_search.o $(OBJ)perf.o

MCTARGETS    = $(OBJ)Box.o $(OBJ)HashGrid.o $(OBJ)FastGrid.o $(OBJ)QueryPoint.o \
               $(OBJ)FastBox.o \
               $(OBJ)ANNInterpolator.o $(OBJ)FastInterpolator.o $(OBJ)TetraBox.o \
               $(OBJ)Tetraeder.o $(OBJ)PlaneInterpolator.o $(OBJ)LSPInterpolator.o 

SHOWTARGETS  = $(OBJ)show.o $(OBJ)camera.o

VIEWTARGETS  = $(OBJ)MoveDock.o $(OBJ)ObjectDialog.o $(OBJ)MatrixDialog.o $(OBJ)MainWindow.o \
               $(OBJ)Viewport.o \
               $(OBJ)EventHandler.o $(OBJ)ObjectHandler.o $(OBJ)RenderFrame.o \
               $(OBJ)ViewerWindow.o $(OBJ)TouchPad.o \
               
IOTARGETS   =  #$(OBJ)fileWriter.o $(OBJ)plyWriter.o $(OBJ)fileReader.o \
               #$(OBJ)plyReader.o $(OBJ)gotoxy.o

all: mcubes viewer 

mcubes: $(OBJ)libnewmat.a $(OBJ)libANN.a $(OBJ)libgsl.a $(LIB3DTARGETS) $(IOTARGETS) $(MCTARGETS)
	@echo -e "\nCompiling and Linking Marching Cubes Main Programm...\n"
	@$(CPP) $(CFLAGS) -o $(BIN)mcubes $(MCSRC)main.cc $(OBJ)libgsl.a $(OBJ)libgslcblas.a $(GLLIBS) $(ANNTARGETS) $(MCTARGETS) $(IOTARGETS) $(LIB3DTARGETS) $(OBJ)libnewmat.a -lgsl 


	
viewer: $(VIEWTARGETS) $(LIB3DTARGETS) $(VIEWSRC)Viewer.cpp
	@echo -e "\nCompiling and Linking Viewer...\n"
	@$(CPP) $(CFLAGS) -o $(BIN)viewer $(VIEWSRC)Viewer.cpp $(LIB3DTARGETS)\
	        $(VIEWTARGETS) $(GLLIBS) $(QTLIBS)

######################################################################
# ----------------------------- I/O ----------------------------------
######################################################################

$(OBJ)fileWriter.o: $(IOSRC)fileWriter.*
	@echo "Compiling File Writer..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)fileWriter.o $(IOSRC)fileWriter.cc

$(OBJ)plyWriter.o: $(IOSRC)plyWriter.*
	@echo "Compiling PLY Writer..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)plyWriter.o $(IOSRC)plyWriter.cc

$(OBJ)fileReader.o: $(IOSRC)fileReader.*
	@echo "Compiling File Writer..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)fileReader.o $(IOSRC)fileReader.cc

$(OBJ)plyReader.o: $(IOSRC)plyReader.*
	@echo "Compiling PLY Reader..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)plyReader.o $(IOSRC)plyReader.cc

$(OBJ)gotoxy.o: $(IOSRC)gotoxy.cc
	@echo "Compiling GotoXY..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)gotoxy.o $(IOSRC)gotoxy.cc

######################################################################
# -------------------------- PRIMITIVES ------------------------------
######################################################################

$(OBJ)baseVertex.o: $(MESHSRC)baseVertex.*
	@echo "Compiling Base Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)baseVertex.o $(MESHSRC)baseVertex.cc

$(OBJ)normal.o: $(OBJ)baseVertex.o $(MESHSRC)normal.*
	@echo "Compiling Normal..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)normal.o $(MESHSRC)normal.cc

$(OBJ)colorVertex.o: $(OBJ)baseVertex.o $(MESHSRC)colorVertex.*
	@echo "Compiling Color Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)colorVertex.o $(MESHSRC)colorVertex.cc

$(OBJ)staticMesh.o: $(MESHSRC)staticMesh.*
	@echo "Compiling Static Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)staticMesh.o $(MESHSRC)staticMesh.cc


######################################################################
# --------------------- MARCHING CUBES CLASSES -----------------------
######################################################################

$(OBJ)Box.o: $(MCSRC)Box.*
	@echo "Compiling Marching Cubes Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Box.o $(MCSRC)Box.cpp

$(OBJ)TetraBox.o: $(MCSRC)TetraederBox.*
	@echo "Compiling Marching Tetraeder Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TetraBox.o $(MCSRC)TetraederBox.cpp

$(OBJ)Tetraeder.o: $(MCSRC)Tetraeder.*
	@echo "Compiling Tetraeder..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Tetraeder.o $(MCSRC)Tetraeder.cpp

$(OBJ)HashGrid.o: $(MCSRC)HashGrid.*
	@echo "Compiling Hash Grid..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)HashGrid.o $(MCSRC)HashGrid.cpp
	
$(OBJ)FastGrid.o: $(MCSRC)FastGrid.*
	@echo "Compiling Fast Grid..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastGrid.o $(MCSRC)FastGrid.cpp
	
$(OBJ)FastBox.o: $(MCSRC)FastBox.cpp
	@echo "Compiling Fast Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastBox.o $(MCSRC)FastBox.cpp
	
$(OBJ)QueryPoint.o: $(MCSRC)QueryPoint.*
	@echo "Compiling Query Point..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)QueryPoint.o $(MCSRC)QueryPoint.cpp

$(OBJ)ANNInterpolator.o: $(MCSRC)ANNInterpolator.*
	@echo "Compiling ANN Interpolator..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ANNInterpolator.o $(MCSRC)ANNInterpolator.cpp

$(OBJ)FastInterpolator.o: $(MCSRC)FastInterpolator.*
	@echo "Compiling Fast Interpolator..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastInterpolator.o $(MCSRC)FastInterpolator.cpp
	
$(OBJ)PlaneInterpolator.o: $(MCSRC)PlaneInterpolator.*
	@echo "Compiling Plane Interpolator..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PlaneInterpolator.o $(MCSRC)PlaneInterpolator.cpp 
	
$(OBJ)LSPInterpolator.o: $(MCSRC)LSPInterpolator.*
	@echo "Compiling LSP Interpolator..."
	@$(CPP) $(CFLAGS) -c -o  $(OBJ)LSPInterpolator.o $(MCSRC)LSPInterpolator.cpp

######################################################################
# -------------------------- ANN LIBRARY -----------------------------
######################################################################

$(OBJ)libANN.a: $(OBJ)ANN.o $(OBJ)brute.o $(OBJ)kd_tree.o \
                $(OBJ)kd_util.o $(OBJ)kd_split.o \
                $(OBJ)kd_search.o $(OBJ)kd_pr_search.o \
                $(OBJ)kd_fix_rad_search.o $(OBJ)kd_dump.o \
                $(OBJ)bd_tree.o $(OBJ)bd_search.o \
                $(OBJ)bd_pr_search.o $(OBJ)bd_fix_rad_search.o \
                $(OBJ)perf.o
	@echo -e "\nLinking ANN Library... \n"
	@$(AR) -c -r -s  $(OBJ)libANN.a $(OBJ)ANN.o $(OBJ)brute.o \
                  	  $(OBJ)kd_tree.o $(OBJ)kd_util.o \
                      $(OBJ)kd_split.o $(OBJ)kd_search.o \
                      $(OBJ)kd_pr_search.o $(OBJ)kd_fix_rad_search.o \
                      $(OBJ)kd_dump.o $(OBJ)bd_tree.o \
                      $(OBJ)bd_search.o $(OBJ)bd_pr_search.o \
                      $(OBJ)bd_fix_rad_search.o $(OBJ)perf.o
	@ranlib $(OBJ)libANN.a


$(OBJ)ANN.o: $(ANNSRC)ANN.cpp
	@echo "Compiling ANN..."
	@$(CPP) -c $(CFLAGS) -o $(OBJ)ANN.o $(ANNSRC)ANN.cpp

$(OBJ)brute.o: $(ANNSRC)brute.cpp
	@echo "Compiling Brute..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)brute.o $(ANNSRC)brute.cpp

$(OBJ)kd_tree.o: $(ANNSRC)kd_tree.cpp
	@echo "Compiling KD-Tree..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_tree.o $(ANNSRC)kd_tree.cpp

$(OBJ)kd_util.o: $(ANNSRC)kd_util.cpp
	@echo "Compiling KD-Utils..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_util.o $(ANNSRC)kd_util.cpp

$(OBJ)kd_split.o: $(ANNSRC)kd_split.cpp
	@echo "Compiling KD-Split..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_split.o $(ANNSRC)kd_split.cpp

$(OBJ)kd_search.o: $(ANNSRC)kd_search.cpp
	@echo "Compiling KD-Search..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_search.o $(ANNSRC)kd_search.cpp

$(OBJ)kd_pr_search.o: $(ANNSRC)kd_pr_search.cpp
	@echo "Compiling KD-PR-Search..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_pr_search.o $(ANNSRC)kd_pr_search.cpp

$(OBJ)kd_fix_rad_search.o: $(ANNSRC)kd_fix_rad_search.cpp
	@echo "Compiling KD-Fixed-Radius-Search..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_fix_rad_search.o $(ANNSRC)kd_fix_rad_search.cpp

$(OBJ)kd_dump.o: $(ANNSRC)kd_dump.cpp
	@echo "Compiling KD-Dump..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_dump.o $(ANNSRC)kd_dump.cpp

$(OBJ)bd_tree.o: $(ANNSRC)bd_tree.cpp
	@echo "Compiling BD-Tree..."
	@$(CPP) -c $(CFLAGS) -o $(OBJ)bd_tree.o $(ANNSRC)bd_tree.cpp

$(OBJ)bd_search.o: $(ANNSRC)bd_search.cpp
	@echo "Compiling BD-Search..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_search.o $(ANNSRC)bd_search.cpp

$(OBJ)bd_pr_search.o: $(ANNSRC)bd_pr_search.cpp
	@echo "Compiling BD-PR-Search..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_pr_search.o $(ANNSRC)bd_pr_search.cpp

$(OBJ)bd_fix_rad_search.o: $(ANNSRC)bd_fix_rad_search.cpp
	@echo "Compiling BD-Fix-Radius..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_fix_rad_search.o $(ANNSRC)bd_fix_rad_search.cpp

$(OBJ)perf.o: $(ANNSRC)perf.cpp
	@echo "Compiling Perf..."
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)perf.o $(ANNSRC)perf.cpp


#############################################################
# GSL LIBRARY
#############################################################	

$(OBJ)libgsl.a:
	cd $(GSLSRC); ./configure --disable-shared
	cd $(GSLSRC); make
	@ranlib $(GSLSRC).libs/libgsl.a
	@ranlib $(GSLSRC)cblas/.libs/libgslcblas.a 
	cp $(GSLSRC).libs/libgsl.a $(OBJ)
	cp $(GSLSRC)cblas/.libs/libgslcblas.a $(OBJ)


#############################################################
# NEWMAT LIBRARY
#############################################################
$(OBJ)libnewmat.a:
	@echo "Compiling NEWMAT library..."
	@cd $(NMSRC); make;
	@mv $(NMSRC)libnewmat.a $(OBJ)
	@ranlib $(OBJ)libnewmat.a
	
#############################################################
# VIEWER
#############################################################

$(OBJ)MainWindow.o: $(VIEWSRC)viewer.ui
	@echo "Compiling Main Window..."
	@uic $(VIEWSRC)viewer.ui > $(VIEWSRC)MainWindow.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MainWindow.o $(VIEWSRC)MainWindow.cpp
	
$(OBJ)MatrixDialog.o: $(VIEWSRC)matrixdialog.ui
	@echo "Compiling Matrix Dialog..."
	@uic $(VIEWSRC)matrixdialog.ui > $(VIEWSRC)MatrixDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MatrixDialog.o $(VIEWSRC)MatrixDialog.cpp
	
$(OBJ)ObjectDialog.o: $(VIEWSRC)objectdialog.ui
	@echo "Compiling Object Selection Dialog"
	@uic $(VIEWSRC)objectdialog.ui > $(VIEWSRC)ObjectDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectDialog.o $(VIEWSRC)ObjectDialog.cpp
	
$(OBJ)MoveDock.o: $(VIEWSRC)movedock.ui
	@echo "Compiling Move Dock"
	@uic $(VIEWSRC)movedock.ui > $(VIEWSRC)MoveDock.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MoveDock.o $(VIEWSRC)MoveDock.cpp
	
$(OBJ)TouchPad.o: $(VIEWSRC)TouchPad.*
	@echo "Compiling Touch Pad..."
	@$(MOC) -i $(VIEWSRC)TouchPad.h > $(VIEWSRC)TouchPadMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TouchPad.o $(VIEWSRC)TouchPad.cpp
	
$(OBJ)ViewerWindow.o: $(VIEWSRC)ViewerWindow.*
	@echo "Compiling Viewer Window"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ViewerWindow.o $(VIEWSRC)ViewerWindow.cpp

$(OBJ)Viewport.o: $(VIEWSRC)Viewport.*
	@echo "Compiling Viewport..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Viewport.o $(VIEWSRC)Viewport.cpp	
	
$(OBJ)EventHandler.o: $(VIEWSRC)EventHandler.*
	@echo "Compiling Event Handler..."	
	@$(MOC) -i $(VIEWSRC)EventHandler.h > $(VIEWSRC)EventHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)EventHandler.o $(VIEWSRC)EventHandler.cpp
	
$(OBJ)ObjectHandler.o: $(VIEWSRC)ObjectHandler.*
	@echo "Compiling Object Handler..."
	@$(MOC) -i $(VIEWSRC)ObjectHandler.h > $(VIEWSRC)ObjectHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectHandler.o $(VIEWSRC)ObjectHandler.cpp
	
$(OBJ)RenderFrame.o: $(VIEWSRC)RenderFrame.*
	@echo "Compiling Render Frame..."
	@$(MOC) -i $(VIEWSRC)RenderFrame.h > $(VIEWSRC)RenderFrameMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)RenderFrame.o $(VIEWSRC)RenderFrame.cpp

#############################################################
# LIB3D
#############################################################

$(OBJ)BaseVertex.o: $(LIB3DSRC)BaseVertex.*
	@echo "Compiling Base Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)BaseVertex.o $(LIB3DSRC)BaseVertex.cpp
	
$(OBJ)BoundingBox.o: $(LIB3DSRC)BoundingBox.*
	@echo "Compiling Bounding Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)BoundingBox.o $(LIB3DSRC)BoundingBox.cpp
	
$(OBJ)ColorVertex.o: $(LIB3DSRC)ColorVertex.*
	@echo "Compiling Color Vertex..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ColorVertex.o $(LIB3DSRC)ColorVertex.cpp

$(OBJ)Normal.o: $(LIB3DSRC)Normal.*
	@echo "Compiling Normal..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Normal.o $(LIB3DSRC)Normal.cpp
	
$(OBJ)Matrix4.o: $(LIB3DSRC)Matrix4.*
	@echo "Compiling 4x4 Matrix..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Matrix4.o $(LIB3DSRC)Matrix4.cpp

$(OBJ)Renderable.o: $(LIB3DSRC)Renderable.*
	@echo "Compiling Renderable..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Renderable.o $(LIB3DSRC)Renderable.cpp
	
$(OBJ)Tube.o: $(LIB3DSRC)Tube.*
	@echo "Compiling Tube..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Tube.o $(LIB3DSRC)Tube.cpp	
	
$(OBJ)GroundPlane.o: $(LIB3DSRC)GroundPlane.*
	@echo "Compiling Ground Plane..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)GroundPlane.o $(LIB3DSRC)GroundPlane.cpp

$(OBJ)CoordinateAxes.o: $(LIB3DSRC)CoordinateAxes.*
	@echo "Compiling Coordinate Axes..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)CoordinateAxes.o $(LIB3DSRC)CoordinateAxes.cpp
	
$(OBJ)Quaternion.o: $(LIB3DSRC)Quaternion.*
	@echo "Compiling Quaternion..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Quaternion.o $(LIB3DSRC)Quaternion.cpp
	
$(OBJ)PointCloud.o: $(LIB3DSRC)PointCloud.*
	@echo "Compiling Point Cloud..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)PointCloud.o $(LIB3DSRC)PointCloud.cpp
	
$(OBJ)NormalCloud.o: $(LIB3DSRC)NormalCloud.*
	@echo "Compiling Normal Point Cloud..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)NormalCloud.o $(LIB3DSRC)NormalCloud.cpp
	
$(OBJ)StaticMesh.o: $(LIB3DSRC)StaticMesh.*
	@echo "Compiling Static Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)StaticMesh.o $(LIB3DSRC)StaticMesh.cpp
	
$(OBJ)TriangleMesh.o: $(LIB3DSRC)TriangleMesh.*
	@echo "Compiling Triangle Mesh..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TriangleMesh.o $(LIB3DSRC)TriangleMesh.cpp
	
clean:
	@echo -e "\nCleaning up...\n"
	@rm -f *.*~
	@rm -f $(OBJ)*.o
	@rm -f $(MCSRC)*.*~
	@rm -f $(MESHSRC)*.*~
	@rm -f $(SHOWSRC)*.*~
	@rm -f $(IOSRC)*.*~

full_clean: clean
	@rm -f $(OBJ)*
	cd $(GSLSRC); make clean
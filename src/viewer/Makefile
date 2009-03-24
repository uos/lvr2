###############################################################################
#
#	MAKEFILE FOR VIEWER LIBRARY
#   -----------------------------------
#
#   Author: Thomas Wiemann
#   Date  : 03/23/2009
#
###############################################################################

include ../../Makefile.options

OBJ = ../../obj/
BIN = ../../bin/
LIB = ../../lib/

VIEWTARGETS  = $(OBJ)MoveDock.o $(OBJ)ObjectDialog.o $(OBJ)MatrixDialog.o $(OBJ)MainWindow.o \
               $(OBJ)Viewport.o \
               $(OBJ)EventHandler.o $(OBJ)ObjectHandler.o $(OBJ)RenderFrame.o \
               $(OBJ)ViewerWindow.o $(OBJ)TouchPad.o

all: libviewer.a

clean:
	@echo -e "Cleaning Viewer Library..."
	@rm -rf $(VIEWTARGETS)
	@rm -rf $(LIB)libviewer.a
	
################################ VIEWER LIBRARY  ##############################

libviewer.a: $(VIEWTARGETS)
	@echo -e "\nLinking Viewer Library...\n"
	@$(AR) -c -r -s $(LIB)libviewer.a $(VIEWTARGETS)
	
	
############################### OBJECT FILES ##################################

$(OBJ)MainWindow.o: viewer.ui
	@echo "Compiling Main Window..."
	@uic viewer.ui > MainWindow.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MainWindow.o MainWindow.cpp

$(OBJ)MatrixDialog.o: matrixdialog.ui
	@echo "Compiling Matrix Dialog..."
	@uic matrixdialog.ui > MatrixDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MatrixDialog.o MatrixDialog.cpp

$(OBJ)ObjectDialog.o: objectdialog.ui
	@echo "Compiling Object Selection Dialog"
	@uic objectdialog.ui > ObjectDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectDialog.o ObjectDialog.cpp

$(OBJ)MoveDock.o: movedock.ui
	@echo "Compiling Move Dock"
	@uic movedock.ui > MoveDock.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MoveDock.o MoveDock.cpp

$(OBJ)TouchPad.o: TouchPad.*
	@echo "Compiling Touch Pad..."
	@$(MOC) -i TouchPad.h > TouchPadMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TouchPad.o TouchPad.cpp

$(OBJ)ViewerWindow.o: ViewerWindow.*
	@echo "Compiling Viewer Window"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ViewerWindow.o ViewerWindow.cpp

$(OBJ)Viewport.o: Viewport.*
	@echo "Compiling Viewport..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Viewport.o Viewport.cpp	

$(OBJ)EventHandler.o: EventHandler.*
	@echo "Compiling Event Handler..."	
	@$(MOC) -i EventHandler.h > EventHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)EventHandler.o EventHandler.cpp

$(OBJ)ObjectHandler.o: ObjectHandler.*
	@echo "Compiling Object Handler..."
	@$(MOC) -i ObjectHandler.h > ObjectHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectHandler.o ObjectHandler.cpp

$(OBJ)RenderFrame.o: RenderFrame.*
	@echo "Compiling Render Frame..."
	@$(MOC) -i RenderFrame.h > RenderFrameMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)RenderFrame.o RenderFrame.cpp


               
              
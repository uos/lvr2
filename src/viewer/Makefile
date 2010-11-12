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

UIC = uic-qt4
MOC = moc-qt4

OBJ = $(BASEDIR)/obj/
LIB = $(BASEDIR)/lib/
BIN = $(BASEDIR)/bin/

VIEWTARGETS  = $(OBJ)MoveDock.o $(OBJ)ObjectDialog.o $(OBJ)MatrixDialog.o $(OBJ)MainWindow.o \
               $(OBJ)Viewport.o \
               $(OBJ)EventHandler.o $(OBJ)ObjectHandler.o $(OBJ)RenderFrame.o \
               $(OBJ)ViewerWindow.o $(OBJ)TouchPad.o

all: libviewer.a $(BIN)viewer

clean:
	@echo "[CLN] Viewer"
	@rm -rf $(VIEWTARGETS)
	@rm -rf $(LIB)libviewer.a

################################ VIEWER LIBRARY  ##############################

libviewer.a: $(VIEWTARGETS)
	@echo "[LIB] Viewer Library"
	@$(AR) -c -r -s $(LIB)libviewer.a $(VIEWTARGETS)
	
$(BIN)viewer: Viewer.cpp
	@echo "[BIN] Viewer"
	@$(CPP) $(CFLAGS) -o $(BIN)viewer Viewer.cpp $(GLLIBS) $(QTLIBS) $(VIEWTARGETS) $(LIB3D)


############################### OBJECT FILES ##################################

$(OBJ)MainWindow.o: viewer.ui
	@echo "[CPP] Main Window"
	@$(UIC) viewer.ui > MainWindow.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MainWindow.o MainWindow.cpp

$(OBJ)MatrixDialog.o: matrixdialog.ui
	@echo "[CPP] Matrix Dialog"
	@$(UIC) matrixdialog.ui > MatrixDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MatrixDialog.o MatrixDialog.cpp

$(OBJ)ObjectDialog.o: objectdialog.ui
	@echo "[CPP] Object Selection Dialog"
	@$(UIC) objectdialog.ui > ObjectDialog.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectDialog.o ObjectDialog.cpp

$(OBJ)MoveDock.o: movedock.ui
	@echo "[CPP] Move Dock"
	@$(UIC) movedock.ui > MoveDock.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)MoveDock.o MoveDock.cpp

$(OBJ)TouchPad.o: TouchPad.*
	@echo "[CPP] Touch Pad"
	@$(MOC) -i TouchPad.h > TouchPadMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TouchPad.o TouchPad.cpp

$(OBJ)ViewerWindow.o: ViewerWindow.*
	@echo "[CPP] Viewer Window"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ViewerWindow.o ViewerWindow.cpp

$(OBJ)Viewport.o: Viewport.*
	@echo "[CPP] Viewport"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Viewport.o Viewport.cpp	

$(OBJ)EventHandler.o: EventHandler.*
	@echo "[CPP] Event Handler"	
	@$(MOC) -i EventHandler.h > EventHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)EventHandler.o EventHandler.cpp

$(OBJ)ObjectHandler.o: ObjectHandler.*
	@echo "[CPP] Object Handler"
	@$(MOC) -i ObjectHandler.h > ObjectHandlerMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ObjectHandler.o ObjectHandler.cpp

$(OBJ)RenderFrame.o: RenderFrame.*
	@echo "[CPP] Render Frame"
	@$(MOC) -i RenderFrame.h > RenderFrameMoc.cpp
	@$(CPP) $(CFLAGS) -c -o $(OBJ)RenderFrame.o RenderFrame.cpp
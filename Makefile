###############################################################################
#
#	MESHING SOFTWARE MAKEFILE
#   -------------------------------------
#
#   Author: Thomas Wiemann
#   Date  : 03/23/2009
#
###############################################################################

include Makefile.options

MCLIBS      = lib/libmc.a lib/libnewmat.a lib/libmodel.a lib/libann.a $(GLLIBS)
VIEWERLIBS  = $(GLLIBS) $(QTLIBS) lib/libviewer.a lib/libmodel.a

MCSRC       = src/libmc/
VIEWERSRC   = src/viewer/


all: lib/libnewmat.a lib/libann.a lib/libmodel.a lib/libmc.a lib/libviewer.a bin/mcubes bin/viewer

lib/libnewmat.a: 
	@echo -e "Building Newmat Library..."
	@cd src/newmat; make -s;
	@mv -f src/newmat/libnewmat.a ./lib/libnewmat.a
	
lib/libann.a:
	@echo -e "\nBuilding ANN Library...\n"
	@cd src/ann; make -s;
	
lib/libmodel.a:
	@echo -e "\nBuilding 3D Modelling Library...\n"
	@cd src/lib3d; make -s;
	
lib/libmc.a:
	@echo -e "\nBuilding Marching Cubes Library...\n"
	@cd src/libmc; make -s
	
lib/libviewer.a:
	@echo -e "\nBuilding Viewer Library...\n"
	@cd src/viewer; make -s

bin/mcubes: $(MCLIBS) 
	@echo -e "\nCompiling and Linking Marching Cubes Programm...\n"
	@$(CPP) $(CFLAGS) -o bin/mcubes $(MCSRC)main.cc $(MCLIBS) 
	
bin/viewer: $(VIEWERLIBS)
	@echo -e "\nCompiling and Linking Viewer...\n"
	@$(CPP) $(CFLAGS) -o bin/viewer $(VIEWERSRC)Viewer.cpp $(VIEWERLIBS)
	
clean:
	@echo -e "\nCleaning up...\n"
	@rm -f lib/*
	@rm -f bin/*
	@cd src/newmat; make -s clean;
	@cd src/ann; make -s clean;
	@cd src/lib3d; make -s clean;
	@cd src/libmc; make -s clean;
	@cd src/viewer; make -s clean;
	
	 
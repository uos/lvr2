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


EXTLIBS	    = $(NEWMAT) $(LIBANN)
MCLIBS      = lib/libmc.a $(EXTLIBS) $(GLLIBS)
VIEWERLIBS  = $(GLLIBS) $(QTLIBS) $(LIB)libviewer.a 

all: dirs $(EXTLIBS) $(BIN)mcubes $(BIN)viewer

dirs:
	@mkdir -p bin
	@mkdir -p lib
	@mkdir -p obj

$(NEWMAT): 
	@echo "[LIB] Newmat"
	@cd $(EXT)newmat; make -s;
	@mv -f $(EXT)newmat/libnewmat.a ./lib/libnewmat.a

$(LIBANN):
	@cd $(EXT)ann; make -s;

$(LIB)libmc.a: 
	@cd $(MCSRC); make -s

$(LIB)libviewer.a: src/viewer/*.*
	@cd $(VIEWERSRC); make -s

$(BIN)mcubes: $(MCLIBS) 
	@echo "[BIN] Marching Cubes Program"
	@$(CPP) $(CFLAGS) $(MCSRC)main.cc -o $(BIN)mcubes $(MCLIBS) $(LIB3D)

$(BIN)viewer: $(VIEWERLIBS) 
	@echo "[BIN] Viewer"
	@$(CPP) $(CFLAGS) -o $(BIN)viewer $(VIEWERSRC)Viewer.cpp $(VIEWERLIBS) $(LIB3D)

clean: clean_mcubes clean_viewer

clean_mcubes:
	@rm -f $(LIB)libmc.a
	@rm -f $(BIN)mcubes
	@cd $(MCSRC); make -s clean;

clean_viewer:
	@rm -f $(BIN)libviewer.a
	@rm -f $(BIN)viewer
	@cd $(VIEWERSRC); make -s clean;

clean_ext:
	@cd $(EXT)newmat; make -s clean;
	@cd $(EXT)ann; make -s clean;
	@rm -f $(LIB)libann.a
	@rm -f $(LIB)libnewmat.a



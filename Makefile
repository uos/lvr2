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


#EXTLIB      = $(NEWMAT) $(LIBANN)
#MCLIBS      = $(LIB)libmc.a $(EXTLIBS) $(GLLIBS)
#VIEWERLIBS  = $(GLLIBS) $(QTLIBS) $(LIB)libviewer.a 

#all: dirs $(EXTLIBS) $(BIN)mcubes $(BIN)viewer

all: dirs ann newmat
	@make -s -C src/mcubes
	@make -s -C src/viewer

dirs:
	@mkdir -p bin
	@mkdir -p lib
	@mkdir -p obj

newmat: $(NEWMAT)
	@echo "[LIB] Newmat"
	@cd $(EXT)newmat; make -s;
	@mv -f $(EXT)newmat/libnewmat.a ./lib/libnewmat.a

ann: $(LIBANN)
	@cd $(EXT)ann; make -s;

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



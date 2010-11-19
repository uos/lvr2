###############################################################################
#
#	MAKEFILE FOR MARCHING CUBES LIBRARY
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

NEWMAT = $(LIB)libnewmat.a
ANN    = $(LIB)libann.a

AR  = ar

MCTARGETS    = $(OBJ)ANNInterpolator.o $(OBJ)FastBox.o \
               $(OBJ)FastGrid.o $(OBJ)Tetraeder.o $(OBJ)TetraBox.o \
               $(OBJ)QueryPoint.o $(OBJ)StannInterpolator.o

CFLAGS+=     -I../../ext/ann/ -I../../ext/stann -I$(HOME)/local/include -fpermissive -g

all: $(BIN)mcubes $(BIN)ascii2ply

clean:

	@echo "[CLN] Marching Cubes"
	@rm -rf $(MCTARGETS)

########################### MARCHING CUBES LIBRARY ############################

$(LIB)libmc.a: $(MCTARGETS)
	@echo  "[LIB] Marching Cubes Library"
	@$(AR) -cr  $(LIB)libmc.a $(MCTARGETS)
	@ranlib  $(LIB)libmc.a

$(BIN)mcubes: $(LIB)libmc.a
	@echo "[BIN] Marching Cubes Program"
	@$(CPP) $(CFLAGS) main.cc -o $(BIN)mcubes $(MCTARGETS) $(GLLIBS) $(LIB3D) $(NEWMAT) $(ANN) $(BOOST)

$(BIN)ascii2ply: ASCIIToPLY.cpp
	@echo "[BIN] ASCII to PLY conversion tool"
	@$(CPP) $(CFLAGS) ASCIIToPLY.cpp -o $(BIN)ascii2ply $(LIB3D) $(GLLIBS)


############################### OBJECT FILES ##################################

$(OBJ)TetraBox.o: TetraederBox.*
	@echo "[CPP] Tetraeder Box"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TetraBox.o TetraederBox.cpp

$(OBJ)Tetraeder.o: Tetraeder.*
	@echo "[CPP] Tetraeder"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Tetraeder.o Tetraeder.cpp

$(OBJ)FastGrid.o: FastGrid.*
	@echo "[CPP] Fast Grid"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastGrid.o FastGrid.cpp

$(OBJ)FastBox.o: FastBox.*
	@echo "[CPP] Marching Cubes Box"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastBox.o FastBox.cpp

$(OBJ)ANNInterpolator.o: ANNInterpolator.*
	@echo "[CPP] ANN Interpolator"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ANNInterpolator.o ANNInterpolator.cpp

$(OBJ)StannInterpolator.o: StannInterpolator.*
	@echo "[CPP] STANN Interpolator"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)StannInterpolator.o StannInterpolator.cpp

$(OBJ)QueryPoint.o: QueryPoint.*
	@echo "[CPP] Query Point"
	@$(CPP) $(CFLAGS) -c -o $(OBJ)QueryPoint.o QueryPoint.cpp


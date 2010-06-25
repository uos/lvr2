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

AR  = ar

MCTARGETS    = $(OBJ)ANNInterpolator.o $(OBJ)FastBox.o \
               $(OBJ)FastGrid.o $(OBJ)Tetraeder.o $(OBJ)TetraBox.o \
               $(OBJ)QueryPoint.o $(OBJ)StannInterpolator.o

CFLAGS+=     -I../../ext/ann/ -I../../ext/stann -I$(HOME)/local/include

all: libmc.a

clean:

	@echo "[CLN] Marching Cubes"
	@rm -rf $(MCTARGETS)

########################### MARCHING CUBES LIBRARY ############################

libmc.a: $(MCTARGETS)
	@echo  "[LIB] Marching Cubes Library"
	$(AR) -cr  $(LIB)libmc.a $(MCTARGETS)
	@ranlib  $(LIB)libmc.a

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

$(OBJ)FastBox.o: FastBox.cpp
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


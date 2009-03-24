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

MCTARGETS    = $(OBJ)ANNInterpolator.o $(OBJ)FastBox.o \
               $(OBJ)FastGrid.o $(OBJ)Tetraeder.o $(OBJ)TetraBox.o \
               $(OBJ)QueryPoint.o $(OBJ)StannInterpolator.o


all: libmc.a

clean:
	@echo Cleaning Marching Cubes object files and library...
	@rm -rf $(MCTARGETS)

########################### MARCHING CUBES LIBRARY ############################

libmc.a: $(MCTARGETS)
	@echo -e "\nLinking Marching Cubes Library...\n"
	$(AR) -cr  $(LIB)libmc.a $(MCTARGETS)
	@ranlib  $(LIB)libmc.a

############################### OBJECT FILES ##################################

$(OBJ)TetraBox.o: TetraederBox.*
	@echo "Compiling Tetraeder Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)TetraBox.o TetraederBox.cpp

$(OBJ)Tetraeder.o: Tetraeder.*
	@echo "Compiling Tetraeder..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)Tetraeder.o Tetraeder.cpp

$(OBJ)FastGrid.o: FastGrid.*
	@echo "Compiling Hash Grid..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastGrid.o FastGrid.cpp

$(OBJ)FastBox.o: FastBox.cpp
	@echo "Compiling Marching Cubes Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)FastBox.o FastBox.cpp

$(OBJ)ANNInterpolator.o: ANNInterpolator.*
	@echo "Compiling ANN Interpolator..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)ANNInterpolator.o ANNInterpolator.cpp

$(OBJ)StannInterpolator.o: StannInterpolator.*
	@echo "Compiling STANN Interpolator..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)StannInterpolator.o StannInterpolator.cpp

$(OBJ)QueryPoint.o: QueryPoint.*
	@echo "Compiling Query Point..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)QueryPoint.o QueryPoint.cpp


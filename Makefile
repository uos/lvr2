CPP = g++
CFLAGS = -O2 -Wall -Wno-write-strings
AR = ar

BIN = bin/
OBJ = obj/

OSFLAGS    = -DMAC_OSX # TO DO: AUTO SELECTION

SHOWSRC    = src/show/
MCSRC      = src/mcubes/
MESHSRC    = src/mesh/
ANNSRC     = src/ann/
IOSRC      = src/io/
GSLSRC     = src/gsl/
NMSRC      = src/newmat/
TESTSRC    = src/test/

#ifndef MAC_OSX
GLLIBS     = -lGL -lGLU -lglut -lgltt -lttf -lgle
#else
GLLIBS     = -framework OpenGL -framework GLUT
#endif

CFLAGS     += -I$(ANNSRC) -I$(GSLSRC)
CFLAGS     += $(OSFLAGS)

ANNTARGETS  = $(OBJ)ANN.o $(OBJ)brute.o $(OBJ)kd_tree.o $(OBJ)kd_util.o \
              $(OBJ)kd_split.o $(OBJ)kd_search.o $(OBJ)kd_pr_search.o \
              $(OBJ)kd_fix_rad_search.o $(OBJ)kd_dump.o $(OBJ)bd_tree.o \
              $(OBJ)bd_search.o $(OBJ)bd_pr_search.o \
              $(OBJ)bd_fix_rad_search.o $(OBJ)perf.o

MCTARGETS   = $(OBJ)baseVertex.o $(OBJ)normal.o $(OBJ)colorVertex.o \
              $(OBJ)staticMesh.o $(OBJ)box.o $(OBJ)distanceFunction.o \
		    $(OBJ)hashGrid.o $(OBJ)tangentPlane.o

SHOWTARGETS = $(OBJ)show.o $(OBJ)camera.o

IOTARGETS   = $(OBJ)fileWriter.o $(OBJ)plyWriter.o $(OBJ)fileReader.o \
              $(OBJ)plyReader.o

all: mcubes show

mcubes: $(OBJ)libnewmat.a $(OBJ)libANN.a $(OBJ)libgsl.a $(IOTARGETS) $(MCTARGETS)
	@echo -e "\nCompiling and Linking Marching Cubes Main Programm..."
	@$(CPP) $(CFLAGS) -o $(BIN)mcubes $(OBJ)libgsl.a $(OBJ)libgslcblas.a \
                          $(GLLIBS) $(ANNTARGETS) $(MCTARGETS) $(IOTARGETS) \
                          $(OBJ)libnewmat.a $(MCSRC)main.cc 
	@echo "DONE."

show: $(SHOWTARGETS)
	@echo -e "\nCompiling and Linking Mesh Viewer..."
	@$(CPP) $(CFLAGS) -o $(BIN)show $(SHOWSRC)main.cc $(SHOWTARGETS) \
                             $(MCTARGETS) $(IOTARGETS) $(GLLIBS) $(ANNTARGETS) $(OBJ)libnewmat.a $(OBJ)libgsl.a $(OBJ)libgslcblas.a
	@echo "DONE."


######################################################################
# --------------------------- FILE I/O -------------------------------
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
# ------------------------------ SHOW --------------------------------
######################################################################

$(OBJ)show.o: $(SHOWSRC)show.*
	@echo "Compiling Show..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)show.o $(SHOWSRC)show.cc

$(OBJ)camera.o: $(SHOWSRC)camera.cc
	@echo "Compiling Camera..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)camera.o $(SHOWSRC)camera.cc

######################################################################
# --------------------- MARCHING CUBES CLASSES -----------------------
######################################################################

$(OBJ)box.o: $(MCSRC)box.*
	@echo "Compiling Marching Cubes Box..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)box.o $(MCSRC)box.cc

$(OBJ)distanceFunction.o: $(MCSRC)distanceFunction.*
	@echo "Compiling Distance Function..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)distanceFunction.o \
                             $(MCSRC)distanceFunction.cc

$(OBJ)tangentPlane.o: $(MCSRC)tangentPlane.*
	@echo "Compiling Tangent Plane..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)tangentPlane.o $(MCSRC)tangentPlane.cc

$(OBJ)hashGrid.o: $(MCSRC)hashGrid.*
	@echo "Compiling Hash Grid..."
	@$(CPP) $(CFLAGS) -c -o $(OBJ)hashGrid.o $(MCSRC)hashGrid.cc

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
	@$(AR) -c -r $(OBJ)libANN.a $(OBJ)ANN.o $(OBJ)brute.o \
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
	cp $(GSLSRC).libs/libgsl.a $(OBJ)
	cp $(GSLSRC)/cblas/.libs/libgslcblas.a $(OBJ)
	@ranlib $(OBJ)libgsl.a
	@ranlib $(OBJ)libgslcblas.a

#############################################################
# NEWMAT LIBRARY
#############################################################
$(OBJ)libnewmat.a:
	@echo "Compiling NEWMAT library..."
	@cd $(NMSRC); make;
	@mv $(NMSRC)libnewmat.a $(OBJ)
	@ranlib $(OBJ)libnewmat.a
clean:
	@echo "Cleaning up..."
	@rm -f *.*~
	@rm -f $(OBJ)*.o
	@rm -f $(MCSRC)*.*~
	@rm -f $(MESHSRC)*.*~
	@rm -f $(SHOWSRC)*.*~
	@rm -f $(IOSRC)*.*~

full_clean: clean
	@rm -f $(OBJ)*
	cd $(GSLSRC); make clean
###############################################################################
#
#	MAKEFILE FOR ANN LIBRARY
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

ANNTARGETS   = $(OBJ)ANN.o $(OBJ)brute.o $(OBJ)kd_tree.o $(OBJ)kd_util.o \
               $(OBJ)kd_split.o $(OBJ)kd_search.o $(OBJ)kd_pr_search.o \
               $(OBJ)kd_fix_rad_search.o $(OBJ)kd_dump.o $(OBJ)bd_tree.o \
               $(OBJ)bd_search.o $(OBJ)bd_pr_search.o \
               $(OBJ)bd_fix_rad_search.o $(OBJ)perf.o

all: libann.a 

clean:
	@echo "[CLN] ANN"
	@rm -rf $(ANNTARGETS)

############################ 3D MODELLING LIBRARY #############################

libann.a: $(ANNTARGETS)
	@echo "[LIB] ANN"
	@$(AR) -c -r -s $(LIB)libann.a $(ANNTARGETS)
	@ranlib  $(LIB)libann.a


############################### OBJECT FILES ##################################

$(OBJ)ANN.o: ANN.cpp
	@echo "[CPP] ANN"
	@$(CPP) -c $(CFLAGS) -o $(OBJ)ANN.o ANN.cpp

$(OBJ)brute.o: brute.cpp
	@echo "[CPP] Brute"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)brute.o brute.cpp

$(OBJ)kd_tree.o: kd_tree.cpp
	@echo "[CPP] KD-Tree"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_tree.o kd_tree.cpp

$(OBJ)kd_util.o: kd_util.cpp
	@echo "[CPP] KD-Utils"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_util.o kd_util.cpp

$(OBJ)kd_split.o: kd_split.cpp
	@echo "[CPP] KD-Split"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_split.o kd_split.cpp

$(OBJ)kd_search.o: kd_search.cpp
	@echo "[CPP] KD-Search"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_search.o kd_search.cpp

$(OBJ)kd_pr_search.o: kd_pr_search.cpp
	@echo "[CPP] KD-PR-Search"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_pr_search.o kd_pr_search.cpp

$(OBJ)kd_fix_rad_search.o: kd_fix_rad_search.cpp
	@echo "[CPP] KD-Fixed-Radius-Search"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_fix_rad_search.o kd_fix_rad_search.cpp

$(OBJ)kd_dump.o: kd_dump.cpp
	@echo "[CPP] KD-Dump"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)kd_dump.o kd_dump.cpp

$(OBJ)bd_tree.o: bd_tree.cpp
	@echo "[CPP] BD-Tree"
	@$(CPP) -c $(CFLAGS) -o $(OBJ)bd_tree.o bd_tree.cpp

$(OBJ)bd_search.o: bd_search.cpp
	@echo "[CPP] BD-Search"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_search.o bd_search.cpp

$(OBJ)bd_pr_search.o: bd_pr_search.cpp
	@echo "[CPP] BD-PR-Search"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_pr_search.o bd_pr_search.cpp

$(OBJ)bd_fix_rad_search.o: bd_fix_rad_search.cpp
	@echo "[CPP] BD-Fix-Radius"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)bd_fix_rad_search.o bd_fix_rad_search.cpp

$(OBJ)perf.o: perf.cpp
	@echo "[CPP] Perf"
	@$(CPP) -c  $(CFLAGS) -o $(OBJ)perf.o perf.cpp
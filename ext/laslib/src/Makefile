# makefile for liblas.a
#
#COPTS    = -g -Wall
COPTS     = -O3 -Wall -Wno-deprecated -DNDEBUG -DUNORDERED
COMPILER  = g++
AR  = ar rc
#BITS     = -64

INCLUDE  = -I/usr/include/ -I../inc -I.

OBJ_LAS = lasreader.o laswriter.o lasreader_las.o lasreader_bin.o lasreader_shp.o lasreader_qfit.o lasreader_txt.o lasreadermerged.o laswriter_las.o laswriter_bin.o laswriter_qfit.o laswriter_txt.o laswaveform13reader.o laswaveform13writer.o lasutility.o lasinterval.o lasspatial.o lasquadtree.o lasindex.o lasfilter.o lastransform.o laszip.o laszipper.o lasunzipper.o lasreadpoint.o lasreaditemcompressed_v1.o lasreaditemcompressed_v2.o laswritepoint.o  laswriteitemcompressed_v1.o laswriteitemcompressed_v2.o integercompressor.o arithmeticdecoder.o arithmeticencoder.o arithmeticmodel.o fopen_compressed.o

all: liblas.a

liblas.a: ${OBJ_LAS}
	$(AR) $@ ${OBJ_LAS}
	cp -p $@ ../lib

.cpp.o:
	${COMPILER} ${BITS} -c ${COPTS} ${INCLUDE} $< -o $@

.c.o: 
	${COMPILER} ${BITS} -c ${COPTS} ${INCLUDE} $< -o $@

clean:
	rm -rf liblas.a
	rm -rf *.o

clobber:
	rm -rf liblas.a
	rm -rf *.o
	rm -rf *~

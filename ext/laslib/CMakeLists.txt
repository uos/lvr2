
find_path(LASLIB_INCLUDE_DIR lasreader.hpp PATH inc)

include_directories(${LASLIB_INCLUDE_DIR})

set(LASLIB_SOURCES
	src/lasreader.cpp
	src/laswriter.cpp
	src/lasreader_las.cpp
	src/lasreader_bin.cpp
	src/lasreader_shp.cpp
	src/lasreader_qfit.cpp
	src/lasreader_txt.cpp
	src/lasreadermerged.cpp
	src/laswriter_las.cpp
	src/laswriter_bin.cpp
	src/laswriter_qfit.cpp
	src/laswriter_txt.cpp
	src/laswaveform13reader.cpp
	src/laswaveform13writer.cpp
	src/lasutility.cpp
	src/lasinterval.cpp
	src/lasspatial.cpp
	src/lasquadtree.cpp
	src/lasindex.cpp
	src/lasfilter.cpp
	src/lastransform.cpp
	src/laszip.cpp
	src/laszipper.cpp
	src/lasunzipper.cpp
	src/lasreadpoint.cpp
	src/lasreaditemcompressed_v1.cpp
	src/lasreaditemcompressed_v2.cpp
	src/laswritepoint.cpp
	src/laswriteitemcompressed_v1.cpp
	src/laswriteitemcompressed_v2.cpp
	src/integercompressor.cpp
	src/arithmeticdecoder.cpp
	src/arithmeticencoder.cpp
	src/arithmeticmodel.cpp
	src/fopen_compressed.cpp)

add_library(lvrlas_static STATIC ${LASLIB_SOURCES})

add_library(lvrlas SHARED ${LASLIB_SOURCES})

install(TARGETS lvrlas_static lvrlas
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lvr/ext/laslib)

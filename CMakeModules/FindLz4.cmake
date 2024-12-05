# - Find LZ4 (lz4.h, liblz4.a, liblz4.so, and liblz4.so.1)
# This module defines
#  LZ4_INCLUDE_DIR, directory containing headers
#  LZ4_LIBS, directory containing lz4 libraries
#  LZ4_STATIC_LIB, path to liblz4.a
#  LZ4_LIBRARY, path to liblz4.so
#  LZ4_FOUND, whether lz4 has been found

find_package(PkgConfig REQUIRED)
pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)
set(LZ4_LIBRARY PkgConfig::lz4)
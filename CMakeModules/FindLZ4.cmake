# - Find LZ4 (lz4.h, liblz4.a, liblz4.so, and liblz4.so.1)
# This module defines
#  LZ4_INCLUDE_DIR, directory containing headers
#  LZ4_LIBS, directory containing lz4 libraries
#  LZ4_STATIC_LIB, path to liblz4.a
#  LZ4_LIBRARY, path to liblz4.so
#  LZ4_FOUND, whether lz4 has been found

set(LZ4_FOUND FALSE)

message(STATUS "Try to find LZ4: find_path + find_library")
find_path(LZ4_INCLUDE_DIR
  NAMES lz4.h
  DOC "lz4 include directory")

find_library(LZ4_LIBRARY
  NAMES lz4 liblz4
  DOC "lz4 library")

if(NOT LZ4_FOUND)
message(STATUS "Try to find LZ4: FindPackageHandleStandardArgs")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZ4
  REQUIRED_VARS LZ4_LIBRARY LZ4_INCLUDE_DIR
  VERSION_VAR LZ4_VERSION)
endif(NOT LZ4_FOUND)

if (LZ4_INCLUDE_DIR)
  file(STRINGS "${LZ4_INCLUDE_DIR}/lz4.h" _lz4_version_lines
    REGEX "#define[ \t]+LZ4_VERSION_(MAJOR|MINOR|RELEASE)")
  string(REGEX REPLACE ".*LZ4_VERSION_MAJOR *\([0-9]*\).*" "\\1" _lz4_version_major "${_lz4_version_lines}")
  string(REGEX REPLACE ".*LZ4_VERSION_MINOR *\([0-9]*\).*" "\\1" _lz4_version_minor "${_lz4_version_lines}")
  string(REGEX REPLACE ".*LZ4_VERSION_RELEASE *\([0-9]*\).*" "\\1" _lz4_version_release "${_lz4_version_lines}")
  set(LZ4_VERSION "${_lz4_version_major}.${_lz4_version_minor}.${_lz4_version_release}")
  unset(_lz4_version_major)
  unset(_lz4_version_minor)
  unset(_lz4_version_release)
  unset(_lz4_version_lines)
endif()

if(NOT LZ4_FOUND)
  find_package(PkgConfig)
  pkg_search_module(LZ4 lz4 liblz4)
  if(TARGET PkgConfig::LZ4)
    set(LZ4_INCLUDE_DIR ${LZ4_INCLUDEDIR})
    set(LZ4_LIBRARY PkgConfig::LZ4)
    set(LZ4_FOUND TRUE)
  endif(TARGET PkgConfig::LZ4)
endif(NOT LZ4_FOUND)

if (LZ4_FOUND)
  set(LZ4_INCLUDE_DIRS "${LZ4_INCLUDE_DIR}")
  set(LZ4_LIBRARIES "${LZ4_LIBRARY}")

  if (NOT TARGET LZ4::LZ4)
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)
    set_target_properties(LZ4::LZ4 PROPERTIES
      IMPORTED_LOCATION "${LZ4_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")
  endif ()
endif ()

mark_as_advanced(LZ4_FOUND)
mark_as_advanced(LZ4_INCLUDE_DIR)
mark_as_advanced(LZ4_LIBRARY)
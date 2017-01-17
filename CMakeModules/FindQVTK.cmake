###############################################################################
# Find QVTK
#
# This sets the following variables:
#
# QVTK_FOUND - True if QVTK was found
# QVTK_INCLUDE_DIR - Directory containing the QVTK include files
# QVTK_LIBRARY - QVTK library
#
# If QVTK_FOUND then QVTK_INCLUDE_DIR is appended to VTK_INCLUDE_DIRS and
# QVTK_LIBRARY is appended to QVTK_LIBRARY_DIR.
#

find_package(VTK)

find_path (QVTK_INCLUDE_DIR QVTKWidget.h HINT ${VTK_INCLUDE_DIRS})

if(VTK_MAJOR_VERSION VERSION_LESS "6")
  find_library (QVTK_LIBRARY QVTK HINTS ${VTK_DIR} ${VTK_DIR}/bin
      PATH_SUFFIXES Release Debug)

  find_package_handle_standard_args(QVTK DEFAULT_MSG
    QVTK_LIBRARY QVTK_INCLUDE_DIR)
else()
  # VTK 6 doesn't have a dedicated QVTK lib anymore
  find_package_handle_standard_args(QVTK DEFAULT_MSG
    QVTK_INCLUDE_DIR)
  set(QVTK_LIBRARY "" CACHE PATH "stubbed QVTK library [obsolete]")
endif()

if(QVTK_FOUND)
  if(VTK_MAJOR_VERSION VERSION_LESS "6")
    get_filename_component (QVTK_LIBRARY_DIR ${QVTK_LIBRARY} PATH)
    set (VTK_LIBRARY_DIRS ${VTK_LIBRARY_DIRS} ${QVTK_LIBRARY_DIR})
  endif()
  set (VTK_INCLUDE_DIRS ${VTK_INCLUDE_DIRS} ${QVTK_INCLUDE_DIR})
endif(QVTK_FOUND)

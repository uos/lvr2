# Try to find Nabo - Nearest neighbour search
# This module will define the following variables:
#   NABO_FOUND              -   indicates whether nabo was found on the system
#   NABO_INCLUDE_DIR        -   the directory for the nabo headerfiles
#   NABO_LIBRARY            -   the compiled nabo library

find_path( NABO_INCLUDE_DIR nabo/nabo.h PATH_SUFFIXES nabo )
find_library( NABO_LIBRARY NAMES nabo libnabo )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NABO DEFAULT_MSG
                                     NABO_LIBRARY NABO_INCLUDE_DIR)

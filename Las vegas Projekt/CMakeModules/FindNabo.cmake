# Try to find Nabo - Nearest neighbour search
# This module will define the following variables:
#   NABO_FOUND              -   indicates whether nabo was found on the system
#   NABO_INCLUDE_DIRS       -   the directories for the nabo headerfiles
#   NABO_LIBRARIES          -   the directory that contains the compiled library

find_path( NABO_INCLUDE_DIR nabo/nabo.h PATH_SUFFIXES nabo )
find_library( NABO_LIBRARY NAMES nabo libnabo )

set( NABO_LIBRARIES ${NABO_LIBRARY} )
set( NABO_INCLUDE_DIRS ${NABO_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nabo DEFAULT_MSG
                                     NABO_LIBRARY NABO_INCLUDE_DIR)

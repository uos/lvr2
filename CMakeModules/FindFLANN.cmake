# Try to find STANN - Nearest neighbour search
# This module will define the following variables:
#   FLANN_FOUND              -   indicates whether FLANN was found on the system
#   FLANN_INCLUDE_DIRS       -   the directories of the FLANN headers

find_path( FLANN_INCLUDE_DIR flann.hpp HINTS "${FLANN_DIR}" PATH_SUFFIXES flann include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLANN DEFAULT_MSG
                                     FLANN_INCLUDE_DIR)

mark_as_advanced( FLANN_INCLUDE_DIR )

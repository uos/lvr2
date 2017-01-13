# Try to find STANN - Nearest neighbour search
# This module will define the following variables:
#   STANN_FOUND              -   indicates whether STANN was found on the system
#   STANN_INCLUDE_DIRS       -   the directories of the STANN headers

find_path( STANN_INCLUDE_DIR sfcnn.hpp HINTS "${STANN_DIR}" PATH_SUFFIXES stann include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STANN DEFAULT_MSG
                                     STANN_INCLUDE_DIR)

mark_as_advanced( STANN_INCLUDE_DIR )

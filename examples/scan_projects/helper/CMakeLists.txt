
set(LVR2_EXAMPLES_SCAN_PROJECTS_SRCS
    src/ScanTypesCompare.cpp
    src/ScanTypesDummies.cpp
    # src/Logging.cpp
)

set(LVR2_EXAMPLES_SCAN_PROJECTS_DEPS
	lvr2_static
	lvr2las_static
	lvr2rply_static
	${LVR2_LIB_DEPENDENCIES}
)

include_directories(include)

add_library(lvr2_examples_scan_projects_helper 
    ${LVR2_EXAMPLES_SCAN_PROJECTS_SRCS})

target_link_libraries(lvr2_examples_scan_projects_helper ${LVR2_EXAMPLES_SCAN_PROJECTS_DEPS} )

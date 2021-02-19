set(LVR2_EXAMPLES_SP_SIMPLE_SRCS
    Main.cpp
)

set(LVR2_EXAMPLES_SP_SIMPLE_DEPS
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
    lvr2_examples_scan_projects_helper
	${LVR2_LIB_DEPENDENCIES}
)

include_directories(include)

add_executable(lvr2_examples_scanprojects_simple
    ${LVR2_EXAMPLES_SP_SIMPLE_SRCS})

target_link_libraries(lvr2_examples_scanprojects_simple
    ${LVR2_EXAMPLES_SP_SIMPLE_DEPS} )

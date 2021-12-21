set(LVR2_EXAMPLES_SP_SCHEMA_SRCS
    Main.cpp
)

set(LVR2_EXAMPLES_SP_SCHEMA_DEPS
	lvr2_static
	lvr2las_static
	lvr2rply_static
    lvr2_examples_scan_projects_helper
	${LVR2_LIB_DEPENDENCIES}
)

add_executable(lvr2_examples_scanprojects_schema
    ${LVR2_EXAMPLES_SP_SCHEMA_SRCS})

target_link_libraries(lvr2_examples_scanprojects_schema
    ${LVR2_EXAMPLES_SP_SCHEMA_DEPS} )

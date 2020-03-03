set(RIEGL_PROJECT_CONVERTER_SRC main.cpp RieglProject.cpp Options.cpp)

add_executable(lvr2_riegl_project_converter ${RIEGL_PROJECT_CONVERTER_SRC})

target_link_libraries(lvr2_riegl_project_converter lvr2_static ${LVR2_LIB_DEPENDENCIES})

install(TARGETS lvr2_riegl_project_converter RUNTIME DESTINATION &{CMAKE_INSTALL_BINDIR})

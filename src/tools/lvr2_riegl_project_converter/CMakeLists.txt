set(RIEGL_PROJECT_CONVERTER main.cpp RieglProject.cpp Options.cpp)

add_executable(riegl_project_converter ${RIEGL_PROJECT_CONVERTER})

target_link_libraries(riegl_project_converter lvr2_static)

install(TARGETS riegl_project_converter RUNTIME DESTINATION &{CMAKE_INSTALL_BINDIR})

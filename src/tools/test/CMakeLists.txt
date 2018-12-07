set(TEST_SRC main.cpp)

add_executable(test ${TEST_SRC})

target_link_libraries(test lvr2_static)

install(TARGETS test RUNTIME DESTINATION &{CMAKE_INSTALL_BINDIR})

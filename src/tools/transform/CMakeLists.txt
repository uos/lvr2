
set(TRANSFORM_SOURCES
    Options.cpp
    Main.cpp
)


add_executable(transform ${TRANSFORM_SOURCES})
set_target_properties(transform PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(transform lvr)
target_link_libraries(transform freenect)
target_link_libraries(transform ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})


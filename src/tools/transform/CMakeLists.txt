
set(TRANSFORM_SOURCES
    Options.cpp
    Main.cpp
)


add_executable(lvr_transform_model ${TRANSFORM_SOURCES})
set_target_properties(lvr_transform_model PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_transform_model lvr)
target_link_libraries(lvr_transform_model freenect)
target_link_libraries(lvr_transform_model ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})


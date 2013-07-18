
set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(reduce ${RECONSTRUCT_SOURCES})
set_target_properties(reduce PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(reduce staticlvr)
target_link_libraries(reduce ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})


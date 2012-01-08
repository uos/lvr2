
set(FILTER_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(filter ${FILTER_SOURCES})

set_target_properties(filter PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
    
target_link_libraries(filter staticlssr)
target_link_libraries(filter ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
    
target_link_libraries(filter ${PCL_LIBRARIES})

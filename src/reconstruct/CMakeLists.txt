# Um das Speicherleck zu finden, alle Main.cpp durch FindLeakMain.cpp ersetzt

set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

message(STATUS "BOOST_FLAG: " ${Boost_SYSTEM_LIBRARY} )

add_executable(findLeakreconstruct ${RECONSTRUCT_SOURCES})
set_target_properties(findLeakreconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(findLeakreconstruct staticlssr)
target_link_libraries(findLeakreconstruct ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(findLeakreconstruct ${OpenMP_CXX_FLAGS})
target_link_libraries(findLeakreconstruct ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(findLeakreconstruct ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

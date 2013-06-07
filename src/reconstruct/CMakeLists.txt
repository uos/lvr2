# Um das Speicherleck zu finden, alle Main.cpp durch FindLeakMain.cpp ersetzt

set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

message(STATUS "BOOST_FLAG: " ${Boost_SYSTEM_LIBRARY} )

add_executable(reconstruct ${RECONSTRUCT_SOURCES})
set_target_properties(reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(reconstruct staticlssr)
target_link_libraries(reconstruct ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(reconstruct ${OpenMP_CXX_FLAGS})
target_link_libraries(reconstruct ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(reconstruct ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

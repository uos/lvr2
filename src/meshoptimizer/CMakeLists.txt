
set(MESH_OPT_SRC
    Options.cpp     
    Main.cpp
)

message(STATUS "BOOST_FLAG: " ${Boost_SYSTEM_LIBRARY} )

add_executable(meshopt ${MESH_OPT_SRC})
set_target_properties(meshopt PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(meshopt staticlssr)
target_link_libraries(meshopt ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(meshopt ${OpenMP_CXX_FLAGS})
target_link_libraries(meshopt ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(meshopt ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

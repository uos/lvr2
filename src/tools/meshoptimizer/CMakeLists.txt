
set(MESH_OPT_SRC
    Options.cpp     
    Main.cpp
)

message(STATUS "BOOST_FLAG: " ${Boost_SYSTEM_LIBRARY} )

add_executable(lvr_mesh_optimizer ${MESH_OPT_SRC})
set_target_properties(lvr_mesh_optimizer PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_mesh_optimizer staticlvr)
target_link_libraries(lvr_mesh_optimizer ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_mesh_optimizer ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_mesh_optimizer ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(lvr_mesh_optimizer ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

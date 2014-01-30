set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_reconstruct ${RECONSTRUCT_SOURCES})
set_target_properties(lvr_reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_reconstruct lvr)
target_link_libraries(lvr_reconstruct freenect)
target_link_libraries(lvr_reconstruct glut)

target_link_libraries(lvr_reconstruct ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_reconstruct ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_reconstruct ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(lvr_reconstruct ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

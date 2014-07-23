set(NORMAL_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_normal_conversion ${NORMAL_SOURCES})
set_target_properties(lvr_normal_conversion PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_normal_conversion lvr)
target_link_libraries(lvr_normal_conversion freenect)
target_link_libraries(lvr_normal_conversion glut)

target_link_libraries(lvr_normal_conversion ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_normal_conversion ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_normal_conversion ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(lvr_normal_conversion ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )


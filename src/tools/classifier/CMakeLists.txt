set(CLASSIFIER_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(classifier ${CLASSIFIER_SOURCES})
set_target_properties(classifier PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(classifier lvr)
target_link_libraries(classifier freenect)
target_link_libraries(classifier glut)

target_link_libraries(classifier ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(classifier ${OpenMP_CXX_FLAGS})
target_link_libraries(classifier ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(classifier ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

set(CLASSIFIER_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_classifier ${CLASSIFIER_SOURCES})
set_target_properties(lvr_classifier PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_classifier lvr)
target_link_libraries(lvr_classifier freenect)
target_link_libraries(lvr_classifier glut)

target_link_libraries(lvr_classifier ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_classifier ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_classifier ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(lvr_classifier ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

INSTALL(TARGETS lvr_classifier
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
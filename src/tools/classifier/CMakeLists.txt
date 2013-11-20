set(CLASSIFIER_SOURCES
	Options.cpp
	Main.cpp
)

add_executable(classifiers ${RECONSTRUCT_SOURCES})
set_target_properties(classifiers PROPERTIES BINARY_OUTPUT_DIRECTORY
	${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(classifiers lvr)
target_link_libraries(classifiers freenect)
target_link_libraries(classifiers glut)

target_link_libraries(classifiers ${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
	${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(classifiers ${OpenMP_CXX_FLAGS})
target_link_libraries(classifiers ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
	target_link_libraries(classifiers ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

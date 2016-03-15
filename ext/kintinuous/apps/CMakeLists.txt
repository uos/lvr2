include_directories(${CMAKE_SOURCE_DIR}/ext/kintinuous/kfusion/include)
link_directories(${LSSR_LIBRARIES} /usr/lib/x86_64-linux-gnu/)

set(LVR_KINFU_SRC
    LVR_KinFu.cpp)
add_executable(lvr_kinfu ${LVR_KINFU_SRC})

target_link_libraries(lvr_kinfu 
	kfusion
	lvr_static
	${OpenCV_LIBS}
	${Boost_SYSTEM_LIBRARY} 
	${Boost_PROGRAM_OPTIONS_LIBRARY}
	${Boost_FILESYSTEM_LIBRARY}
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	)


set_target_properties(lvr_kinfu PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS lvr_kinfu RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${srcs} DESTINATION app COMPONENT main)

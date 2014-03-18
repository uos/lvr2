set(REGISTRATION_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_registration ${REGISTRATION_SOURCES})
set_target_properties(lvr_registration PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_registration lvr)
target_link_libraries(lvr_registration freenect)
target_link_libraries(lvr_registration glut)

target_link_libraries(lvr_registration ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_registration ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_registration ${PCL_LIBRARIES})
if( ${NABO_FOUND} )
    target_link_libraries(lvr_registration ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

INSTALL(TARGETS lvr_registration
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
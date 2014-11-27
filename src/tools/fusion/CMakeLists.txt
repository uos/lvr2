set(FUSION_SOURCES  
    Options.cpp
    Main.cpp
)

add_executable(lvr_fusion ${FUSION_SOURCES})

set_target_properties(lvr_fusion PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(lvr_fusion staticlvr)
target_link_libraries(lvr_fusion ${Boost_FILESYSTEM_LIBRARY}
							 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
							 ${Boost_PROGRAM_OPTIONS_LIBRARY} ${GMP_LIBRARIES}
							 ${MPFR_LIBRARIES} ${CGAL_LIBRARY})
							 
INSTALL(TARGETS lvr_fusion
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

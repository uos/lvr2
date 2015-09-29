include("${CMAKE_CURRENT_LIST_DIR}/lvrTargets.cmake")

option(lvr_USE_STATIC_LIBS OFF)

if(lvr_USE_STATIC_LIBS)
  set(LVR_LIBRARIES lvr::lvr_static)
else()
  set(LVR_LIBRARIES lvr::lvr)
endif()

# Actually, these two variables should not be used because
# target_link_libraries(abc ${LVR_LIBRARIES}) automagically
# correctly includes all INTERFACE_* properties already
get_target_property(LVR_INCLUDE_DIRS ${LVR_LIBRARIES} INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LVR_DEFINITIONS ${LVR_LIBRARIES} INTERFACE_COMPILE_DEFINITIONS)
if(NOT LVR_DEFINITIONS)
  set(LVR_DEFINITIONS)
endif()

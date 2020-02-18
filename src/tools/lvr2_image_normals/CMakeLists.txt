#####################################################################################
# Set source files
#####################################################################################

set(LVR2_IMG_NORMAL_SRC
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_IMG_NORMAL_DEPS
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
	)

if(PCL_FOUND)
  set(LVR2_IMG_NORMAL_DEPS  ${LVR2_IMG_NORMAL_DEPS} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

if( ${NABO_FOUND} )
 set(LVR2_IMG_NORMAL_DEPS  ${LVR2_IMG_NORMAL_DEPS} ${NABO_LIBRARY} )
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_image_normals ${LVR2_IMG_NORMAL_SRC})
target_link_libraries(lvr2_image_normals ${LVR2_IMG_NORMAL_DEPS})

install(TARGETS lvr2_image_normals
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#####################################################################################
# Set source files
#####################################################################################

set(BENCHMARK_SRC
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_BENCHMARK_DEPENDENCIES
    lvr2_static
    lvr2las_static
    lvr2rply_static
    ${LVR2_LIB_DEPENDENCIES}
)

if(CUDA_FOUND)
  list(APPEND LVR2_BENCHMARK_DEPENDENCIES lvr2cuda_static)
endif(CUDA_FOUND)

if( ${NABO_FOUND} )
  list(APPEND LVR2_BENCHMARK_DEPENDENCIES ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################
add_executable(benchmark_knn_normals ${BENCHMARK_SRC})
target_link_libraries(benchmark_knn_normals ${LVR2_BENCHMARK_DEPENDENCIES})

install(TARGETS benchmark_knn_normals
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
add_executable(seg
        LargeScaleReconstruction.cpp
        NodeData.cpp LargeScaleOctree.cpp
        Options.cpp
        DuplicateRemover.cpp
        SparseMatrix.cpp
        SparseVector.cpp
        LineReader.cpp)
add_executable(lvrSlamConverter SlamImporter.cpp)
add_executable(lvr_removeDuplicates removeDuplicates.cpp DuplicateRemover.cpp)
add_executable(lvr_largescale_reconstruct bigGridTest.cpp BigGrid.cpp BigGridKdTree.cpp LineReader.cpp  LargeScaleOptions.cpp)
add_executable(lvr_mpi_largescale_reconstruct LargeScaleReconstruction2.cpp  LargeScaleOptions.cpp BigGrid.cpp BigGridKdTree.cpp LineReader.cpp )
target_link_libraries(lvr_largescale_reconstruct ${Boost_LIBRARIES} lvr )
target_link_libraries(lvr_removeDuplicates  lvr_static )
target_link_libraries(seg  ${Boost_LIBRARIES} ${MPI_LIBRARIES} lvr_static )
target_link_libraries(lvr_mpi_largescale_reconstruct lvr ${Boost_LIBRARIES} ${MPI_LIBRARIES} )

if(CUDA_FOUND)
        target_link_libraries(lvr_largescale_reconstruct lvrcuda)
        target_compile_definitions(lvr_largescale_reconstruct PRIVATE CUDA_FOUND=1)
elseif(OPENCL_FOUND)
        target_compile_definitions(lvr_largescale_reconstruct PRIVATE OPENCL_FOUND=1)
endif()
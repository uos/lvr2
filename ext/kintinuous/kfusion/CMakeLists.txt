set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
add_module_library(kfusion)
target_link_libraries(kfusion 
	${OpenCV_LIBS}
	${CUDA_LIBRARIES} 
	${CUDA_CUDA_LIBRARY} 
	${OPENNI2_LIBRARIES}
	)

#add_executable(pipeline src/main.cpp)
#target_link_libraries(pipeline ${Boost_LIBRARIES})

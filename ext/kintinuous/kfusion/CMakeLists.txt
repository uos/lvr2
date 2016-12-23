set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ftz=true;--prec-div=false;--prec-sqrt=false")
#add_module_library(kfusion)

set(KFUSION_SRC 
	src/capture.cpp
	src/core.cpp
	src/FusionStage.cpp
	src/GridStage.cpp
	src/cyclical_buffer.cpp
	src/device_memory.cpp
	src/imgproc.cpp
	src/LVRPipeline.cpp
	src/kinfu.cpp
	src/MeshStage.cpp
	src/OptimizeStage.cpp
	src/PointSave.cpp
	src/precomp.cpp
	src/projective_icp.cpp
	src/tsdf_volume.cpp
	)
	
set(KFUSION_CUDA_SRC
	src/cuda/imgproc.cu
	src/cuda/proj_icp.cu
	src/cuda/tsdf_volume.cu
	)
	
	
set(LVR_KINFU_SRC
	src/app/LVR_KinFu.cpp
	src/app/Options.cpp
	)

cuda_add_library(kfusion ${KFUSION_SRC} ${KFUSION_CUDA_SRC})

target_link_libraries(kfusion 
	${OpenCV_LIBS}
	${CUDA_LIBRARIES} 
	${CUDA_CUDA_LIBRARY} 
	${OPENNI2_LIBRARIES}
	)
	
if(OPENCV_VIZ_FOUND)

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
endif()


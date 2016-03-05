include_directories(src include) 

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ftz=true;--prec-div=false;--prec-sqrt=false")

set(KF_CUDA_SRC
    src/cuda/imgproc.cu
    src/cuda/proj_icp.cu
    src/cuda/tsdf_volume.cu
    )
    
set(KF_CORE_LIB_SRC
    src/capture.cpp
    src/core.cpp
    src/cyclical_buffer.cpp
    src/device_memory.cpp
    src/imgproc.cpp
    src/kinfu.cpp
    src/precomp.cpp
    src/projective_icp.cpp
    src/tsdf_volume.cpp
    )
    
set(KF_LVR_LIB_SRC
    src/FusionStage.cpp
    src/GridStage.cpp
    src/LVRPipeline.cpp
    src/MeshStage.cpp
    src/OptimizeStage.cpp
    src/Options.cpp
    src/PointSave.cpp
    )
    
cuda_add_library(kfusion ${KF_CUDA_SRC} ${KF_CORE_LIB_SRC} ${KF_LVR_LIB_SRC})

target_link_libraries(kfusion 
	${OpenCV_LIBS}
	${CUDA_LIBRARIES} 
	${CUDA_CUDA_LIBRARY} 
	${OPENNI2_LIBRARIES}
	)

#add_executable(pipeline src/main.cpp)
#target_link_libraries(pipeline ${Boost_LIBRARIES})

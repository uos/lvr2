namespace lvr2 {

template <typename BaseVecT>
CLRaycaster<BaseVecT>::CLRaycaster(const MeshBufferPtr mesh)
:BVHRaycaster<BaseVecT>(mesh)
{ 
    std::cout << "CL Raycaster. Loading OpenCl Kernels..." << std::endl;

    try {
        initOpenCL();
        initOpenCLTreeBuffer();
        copyBVHToGPU();
        createKernel();
        
    }
    catch (cl::Error err)
    {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
    }
    std::cout << "BVH tree copied to gpu." << std::endl;
}

/// PUBLIC FUNTIONS
/// Overload functions ///

template <typename BaseVecT>
bool CLRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction,
    Point<BaseVecT>& intersection
) 
{
    // Cast one ray from one origin
    bool success = false;

    Point<BaseVecT> dst = {0.0, 0.0, 0.0};

    // yeah
    const float* origin_f = reinterpret_cast<const float*>(&origin.x);
    const float* direction_f = reinterpret_cast<const float*>(&direction.x);

    std::vector<float> intersections(3);
    std::vector<uint8_t> hits(1);
    
    try {
        initOpenCLRayBuffer(3,3);
        copyRayDataToGPU(origin_f, 3, direction_f, 3);
        
        m_kernel_one_one.setArg(0, m_rayOriginBuffer);
        m_kernel_one_one.setArg(1, m_rayBuffer);
        m_kernel_one_one.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_one_one.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_one_one.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_one_one.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_one_one.setArg(6, m_resultBuffer);
        m_kernel_one_one.setArg(7, m_resultHitsBuffer);
        
        auto start = std::chrono::steady_clock::now();

        std::cout << "enqueue ND Range kernel..." << std::endl;
        cl::Event evt;
        m_queue.enqueueNDRangeKernel(
            m_kernel_one_one,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange,
            nullptr,
            &evt
        );
        m_queue.finish();
        std::cout << "enqueue ND Range kernel finished" << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now() - start);

        std::cout << "Kernel execution time (ms): " << duration.count() * 1e-6 << std::endl;
        
        m_queue.enqueueReadBuffer(
            m_resultBuffer,
            CL_TRUE,
            0,
            sizeof(float) * 3 * 1,
            intersections.data()
        );
        
        m_queue.enqueueReadBuffer(
            m_resultHitsBuffer,
            CL_TRUE,
            0,
            sizeof(uint8_t) * 1,
            hits.data()
        );
        m_queue.finish();
    }
    catch (cl::Error err)
    {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
    }

    intersection.x = intersections[0];
    intersection.y = intersections[1];
    intersection.z = intersections[2];

    success = hits[0];

    return success;
}

template <typename BaseVecT>
void CLRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    // Cast multiple rays from one origin
    hits.resize(directions.size());
    intersections.resize(directions.size());

    // copy data
    const float* origin_f = reinterpret_cast<const float*>(&origin.x);
    const float* direction_f = reinterpret_cast<const float*>(directions.data());

    

    try { 
        initOpenCLRayBuffer(3, directions.size()*3);
        copyRayDataToGPU(origin_f, 3, direction_f, directions.size()*3);

        m_kernel_one_multi.setArg(0, m_rayOriginBuffer);
        m_kernel_one_multi.setArg(1, m_rayBuffer);
        m_kernel_one_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_one_multi.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_one_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_one_multi.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_one_multi.setArg(6, m_resultBuffer);
        m_kernel_one_multi.setArg(7, m_resultHitsBuffer);

        auto start = std::chrono::steady_clock::now();

        std::cout << "enqueue ND Range kernel..." << std::endl;
        cl::Event evt;
        m_queue.enqueueNDRangeKernel(
            m_kernel_one_multi,
            cl::NullRange,
            cl::NDRange(directions.size()),
            cl::NullRange,
            nullptr,
            &evt
        );
        m_queue.finish();
        std::cout << "enqueue ND Range kernel finished" << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now() - start);

        std::cout << "Kernel execution time (ms): " << duration.count() * 1e-6 << std::endl;
        
        m_queue.enqueueReadBuffer(
            m_resultBuffer,
            CL_TRUE,
            0,
            sizeof(float) * 3 * directions.size(),
            intersections.data()
        );
        
        m_queue.enqueueReadBuffer(
            m_resultHitsBuffer,
            CL_TRUE,
            0,
            sizeof(uint8_t) * directions.size(),
            hits.data()
        );
        m_queue.finish();

    }
    catch (cl::Error err)
    {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
    }

}

template <typename BaseVecT>
void CLRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    // Cast multiple rays from multiple origins

    hits.resize(directions.size());
    intersections.resize(directions.size());

    // copy data
    const float* origin_f = reinterpret_cast<const float*>(origins.data());
    const float* direction_f = reinterpret_cast<const float*>(directions.data());

    try {
        initOpenCLRayBuffer(origins.size() * 3, directions.size()*3);
        copyRayDataToGPU(origin_f, origins.size()*3, direction_f, directions.size()*3);
    
        m_kernel_multi_multi.setArg(0, m_rayOriginBuffer);
        m_kernel_multi_multi.setArg(1, m_rayBuffer);
        m_kernel_multi_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_multi_multi.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_multi_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_multi_multi.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_multi_multi.setArg(6, m_resultBuffer);
        m_kernel_multi_multi.setArg(7, m_resultHitsBuffer);

        auto start = std::chrono::steady_clock::now();

        std::cout << "enqueue ND Range kernel..." << std::endl;
        cl::Event evt;
        m_queue.enqueueNDRangeKernel(
            m_kernel_multi_multi,
            cl::NullRange,
            cl::NDRange(directions.size()),
            cl::NullRange,
            nullptr,
            &evt
        );
        m_queue.finish();
        std::cout << "enqueue ND Range kernel finished" << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now() - start);

        std::cout << "Kernel execution time (ms): " << duration.count() * 1e-6 << std::endl;
        
        m_queue.enqueueReadBuffer(
            m_resultBuffer,
            CL_TRUE,
            0,
            sizeof(float) * 3 * directions.size(),
            intersections.data()
        );
        
        m_queue.enqueueReadBuffer(
            m_resultHitsBuffer,
            CL_TRUE,
            0,
            sizeof(uint8_t) * directions.size(),
            hits.data()
        );
        m_queue.finish();
    }
    catch (cl::Error err)
    {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
    }
    
}

// PRIVATE FUNCTIONS


template<typename BaseVecT>
void CLRaycaster<BaseVecT>::initOpenCL()
{
    std::cout << "Get platforms" << std::endl;
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto const& platform: platforms)
    {
        std::cout << "Found platform: " 
            << platform.getInfo<CL_PLATFORM_NAME>().c_str() 
            << std::endl;
    }
    std::cout << std::endl;

    vector<cl::Device> consideredDevices;
    for (auto const& platform: platforms)
    {
        std::cout << "Get devices of " << platform.getInfo<CL_PLATFORM_NAME>().c_str() << ": " << std::endl;
        cl_context_properties properties[] =
            {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)(platform)(),
                0
            };
        auto tmpContext = cl::Context(CL_DEVICE_TYPE_ALL, properties);
        vector<cl::Device> devices = tmpContext.getInfo<CL_CONTEXT_DEVICES>();
        for (auto const& device : devices)
        {
            std::cout << "Found device: " << device.getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
            std::cout << "Device work units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "Device work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

            // find all devices that support at least OpenCL version 2.0
            //if ((cl::detail::getVersion(device.getInfo<CL_DEVICE_VERSION>().c_str()) >> 16) >= 2)
            //{
                consideredDevices.push_back(device);
            //}
        }
    }
    std::cout << std::endl;

    // preferably choose the first compatible device of type GPU
    bool deviceFound = false;
    for (auto const& device : consideredDevices)
    {
        //ROS_INFO(
        //    "device type: %s device.getInfo<CL_DEVICE_TYPE>() and %s CL_DEVICE_TYPE_GPU",
        //    device.getInfo<CL_DEVICE_TYPE>(), CL_DEVICE_TYPE_GPU
        //);
        if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
        {
            m_device = device;
            m_platform = device.getInfo<CL_DEVICE_PLATFORM>();
            deviceFound = true;
            break;
        }
    }
    if (!deviceFound && consideredDevices.size() > 0)
    {
        // if no device of type GPU was found, choose the first compatible device
        m_device = consideredDevices[0];
        m_platform = m_device.getInfo<CL_DEVICE_PLATFORM>();
        deviceFound = true;
    }
    if (!deviceFound)
    {
        // panic if no compatible device was found
        std::cerr << "No device with compatible OpenCL version found (minimum 2.0)" << std::endl;
    }

    cl_context_properties properties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(m_platform)(),
            0
        };
    m_context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

    std::cout << "Using device " << m_device.getInfo<CL_DEVICE_NAME>().c_str()
            << " of platform " << m_platform.getInfo<CL_PLATFORM_NAME>().c_str()
            << std::endl << std::endl;

    // read kernel file
    std::string cast_rays_kernel(CAST_RAYS_BVH_PROGRAM);

    cl::Program::Sources sources(1, {cast_rays_kernel.c_str(), cast_rays_kernel.length()});

    m_program = cl::Program(m_context, sources);
    try
    {
        m_program.build({m_device});
    }
    catch(cl::Error& err)
    {
        std::cerr << "Error building: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str() << std::endl;
    }

    m_queue = cl::CommandQueue(m_context, m_device, 0);
}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::initOpenCLTreeBuffer()
{

    // create buffers on the device
    m_bvhIndicesOrTriListsBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * BVHRaycaster<BaseVecT>::m_bvh.getIndexesOrTrilists().size()
    );
    m_bvhLimitsnBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * BVHRaycaster<BaseVecT>::m_bvh.getLimits().size()
    );
    m_bvhTriangleIntersectionDataBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * BVHRaycaster<BaseVecT>::m_bvh.getTrianglesIntersectionData().size()
    );
    m_bvhTriIdxListBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * BVHRaycaster<BaseVecT>::m_bvh.getTriIndexList().size()
    );

}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::initOpenCLRayBuffer(int num_origins, int num_rays)
{
    // input buffer

    m_rayOriginBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * num_origins
    );

    m_rayBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * num_rays
    );

    // output buffer
    m_resultBuffer = cl::Buffer(
        m_context,
        CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
        sizeof(float) * num_rays
    );

    m_resultHitsBuffer = cl::Buffer(
        m_context,
        CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
        sizeof(uint8_t) * num_rays/3
    );

}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::copyBVHToGPU()
{
    m_queue.enqueueWriteBuffer(
        m_bvhIndicesOrTriListsBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * BVHRaycaster<BaseVecT>::m_bvh.getIndexesOrTrilists().size(),
        BVHRaycaster<BaseVecT>::m_bvh.getIndexesOrTrilists().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhLimitsnBuffer,
        CL_TRUE,
        0,
        sizeof(float) * BVHRaycaster<BaseVecT>::m_bvh.getLimits().size(),
        BVHRaycaster<BaseVecT>::m_bvh.getLimits().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriangleIntersectionDataBuffer,
        CL_TRUE,
        0,
        sizeof(float) * BVHRaycaster<BaseVecT>::m_bvh.getTrianglesIntersectionData().size(),
        BVHRaycaster<BaseVecT>::m_bvh.getTrianglesIntersectionData().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriIdxListBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * BVHRaycaster<BaseVecT>::m_bvh.getTriIndexList().size(),
        BVHRaycaster<BaseVecT>::m_bvh.getTriIndexList().data()
    );

}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::createKernel()
{
    // one origin one ray

    m_kernel_one_one = cl::Kernel(m_program, "cast_rays_one_one");

    // one origin multiple rays

    m_kernel_one_multi = cl::Kernel(m_program, "cast_rays_one_multi");

    // multiple origins multiple rays

    m_kernel_multi_multi = cl::Kernel(m_program, "cast_rays_multi_multi");

    // test kernel

    m_kernel_test = cl::Kernel(m_program, "test");
}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::copyRayDataToGPU(
    const vector<float>& origins,
    const vector<float>& rays
)
{
    copyRayDataToGPU(origins.data(), origins.size(), rays.data(), rays.size());
}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::copyRayDataToGPU(
    const float* origin_buffer, size_t origin_buffer_size,
    const float* ray_buffer, size_t ray_buffer_size
)
{
    std::cout << "Number of origins: " << origin_buffer_size/3 << endl;
    m_queue.enqueueWriteBuffer(m_rayOriginBuffer, CL_TRUE, 0, sizeof(float) * origin_buffer_size, origin_buffer);

    std::cout << "Size of rays: " << ray_buffer_size/3 << endl;

    std::cout << ray_buffer[297] << std::endl;
    m_queue.enqueueWriteBuffer(m_rayBuffer, CL_TRUE, 0, sizeof(float) * ray_buffer_size, ray_buffer);
}


} // namespace lvr2
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
}

/// PUBLIC FUNTIONS
/// Overload functions ///

template <typename BaseVecT>
Point<BaseVecT> CLRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction
) 
{
    // Cast one ray from one origin

    Point<BaseVecT> dst = {0.0, 0.0, 0.0};

    std::cout << direction << std::endl;

    // yeah
    const float* origin_f = reinterpret_cast<const float*>(&origin.x);
    const float* direction_f = reinterpret_cast<const float*>(&direction.x);

    copyRayDataToGPU(origin_f, 3, direction_f, 3);

    // float *test = reinterpret_cast<float*>(lvr_vec.data());
    // std::cout << test[0] << std::endl;


    // TODO

    return dst;
}

template <typename BaseVecT>
std::vector<Point<BaseVecT> > CLRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions
)
{
    // Cast multiple rays from one origin
    std::vector<Point<BaseVecT> > dst;


    // copy data
    const float* origin_f = reinterpret_cast<const float*>(&origin.x);
    const float* direction_f = reinterpret_cast<float*>(directions.data());
    copyRayDataToGPU(origin_f, 3, direction_f, directions.size()*3);

    // TODO
    return dst;
}

template <typename BaseVecT>
std::vector<Point<BaseVecT> > CLRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions
)
{
    // Cast multiple rays from multiple origins
    std::vector<Point<BaseVecT> > dst;

    // copy data
    const float* origin_f = reinterpret_cast<float*>(origins.data());
    const float* direction_f = reinterpret_cast<float*>(directions.data());
    copyRayDataToGPU(origin_f, origins.size()*3, direction_f, directions.size()*3);

    // TODO
    return dst;
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
    // ifstream in(ros::package::getPath("geometric_localization") + CL_RAY_CAST_KERNEL_FILE);
    // std::string cast_rays_kernel(static_cast<stringstream const&>(stringstream() << in.rdbuf()).str());

    std::string cast_rays_kernel(CAST_RAYS_BVH_PROGRAM);
    

    // ROS_INFO("Got kernel: %s", cast_rays_kernel.c_str());
    // ROS_INFO("Got kernel: %s", CL_RAY_CAST_KERNEL_FILE);

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


    // maybe put this at the very end?

    // create queue to which we will push commands for the device.
    
    // cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

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
    m_rayBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * 3 * num_rays
    );

    m_rayOriginBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * 3 * num_origins
    );

    // output buffer
    m_resultBuffer = cl::Buffer(
        m_context,
        CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
        sizeof(float) * num_rays * 3
    );

    m_resultHitsBuffer = cl::Buffer(
        m_context,
        CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
        sizeof(uint8_t) * num_rays * 1024
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

    m_kernel_one_one.setArg(0, m_rayOriginBuffer);
    m_kernel_one_one.setArg(1, m_rayBuffer);
    m_kernel_one_one.setArg(2, m_bvhIndicesOrTriListsBuffer);
    m_kernel_one_one.setArg(3, m_bvhLimitsnBuffer);
    m_kernel_one_one.setArg(4, m_bvhTriangleIntersectionDataBuffer);
    m_kernel_one_one.setArg(5, m_bvhTriIdxListBuffer);
    m_kernel_one_one.setArg(6, m_resultBuffer);
    m_kernel_one_one.setArg(7, m_resultHitsBuffer);

    // one origin multiple rays

    m_kernel_one_multi = cl::Kernel(m_program, "cast_rays_one_multi");

    m_kernel_one_multi.setArg(0, m_rayOriginBuffer);
    m_kernel_one_multi.setArg(1, m_rayBuffer);
    m_kernel_one_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
    m_kernel_one_multi.setArg(3, m_bvhLimitsnBuffer);
    m_kernel_one_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
    m_kernel_one_multi.setArg(5, m_bvhTriIdxListBuffer);
    m_kernel_one_multi.setArg(6, m_resultBuffer);
    m_kernel_one_multi.setArg(7, m_resultHitsBuffer);

    // multiple origins multiple rays

    m_kernel_multi_multi = cl::Kernel(m_program, "cast_rays_multi_multi");

    m_kernel_multi_multi.setArg(0, m_rayOriginBuffer);
    m_kernel_multi_multi.setArg(1, m_rayBuffer);
    m_kernel_multi_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
    m_kernel_multi_multi.setArg(3, m_bvhLimitsnBuffer);
    m_kernel_multi_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
    m_kernel_multi_multi.setArg(5, m_bvhTriIdxListBuffer);
    m_kernel_multi_multi.setArg(6, m_resultBuffer);
    m_kernel_multi_multi.setArg(7, m_resultHitsBuffer);

}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::copyRayDataToGPU(
    const vector<float>& origins,
    const vector<float>& rays
)
{
    
    std::cout << "Number of origins: " << origins.size()/3 << endl;
    m_queue.enqueueWriteBuffer(m_rayOriginBuffer, CL_TRUE, 0, sizeof(float) * origins.size(), origins.data());

    std::cout << "Size of rays: " << rays.size()/3 << endl;
    m_queue.enqueueWriteBuffer(m_rayBuffer, CL_TRUE, 0, sizeof(float) * rays.size(), rays.data());

}

template<typename BaseVecT>
void CLRaycaster<BaseVecT>::copyRayDataToGPU(
    const float* origins, size_t num_origins,
    const float* rays, size_t num_rays
)
{
    // TODO
}


} // namespace lvr2
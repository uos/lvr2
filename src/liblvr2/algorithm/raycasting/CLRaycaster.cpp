/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "lvr2/algorithm/raycasting/CLRaycaster.hpp"

constexpr char CAST_RAYS_BVH_PROGRAM[] =
    #include "opencl/cast_rays_bvh.cl"
;

namespace lvr2 
{

CLRaycaster::CLRaycaster(const MeshBufferPtr mesh)
:BVHRaycaster(mesh)
,m_warp_size(32)
{ 
    try {
        initOpenCL();
        getDeviceInformation();
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
bool CLRaycaster::castRay(
    const Vector3f& origin,
    const Vector3f& direction,
    Vector3f& intersection
) 
{
    // Cast one ray from one origin
    bool success = false;

    // yeah
    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
    const float* direction_f = reinterpret_cast<const float*>(&direction.coeffRef(0));

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

    intersection.x() = intersections[0];
    intersection.y() = intersections[1];
    intersection.z() = intersections[2];

    success = hits[0];

    return success;
}

void CLRaycaster::castRays(
    const Vector3f& origin,
    const std::vector<Vector3f >& directions,
    std::vector<Vector3f >& intersections,
    std::vector<uint8_t>& hits
)
{
    // Cast multiple rays from one origin
    hits.resize(directions.size());
    intersections.resize(directions.size());

    // copy data
    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
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
        // std::cout << "enqueue ND Range kernel finished" << std::endl;

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

CLRaycasterRuntimeStats CLRaycaster::castRaysWithStats(
    const Vector3f& origin,
    const std::vector<Vector3f >& directions,
    std::vector<Vector3f >& intersections,
    std::vector<uint8_t>& hits
)
{
    CLRaycasterRuntimeStats stats;
    // Cast multiple rays from one origin
    hits.resize(directions.size());
    intersections.resize(directions.size());

    // copy data
    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
    const float* direction_f = reinterpret_cast<const float*>(directions.data());

    std::chrono::time_point<std::chrono::steady_clock> start, end;


    try { 
        start = std::chrono::steady_clock::now();
        initOpenCLRayBuffer(3, directions.size()*3);
        copyRayDataToGPU(origin_f, 3, direction_f, directions.size()*3);
        end = std::chrono::steady_clock::now();
        stats.copy_data_to_device 
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        m_kernel_one_multi.setArg(0, m_rayOriginBuffer);
        m_kernel_one_multi.setArg(1, m_rayBuffer);
        m_kernel_one_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_one_multi.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_one_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_one_multi.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_one_multi.setArg(6, m_resultBuffer);
        m_kernel_one_multi.setArg(7, m_resultHitsBuffer);

        end = std::chrono::steady_clock::now();
        stats.kernel_building 
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
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
        end = std::chrono::steady_clock::now();
        stats.kernel_execution 
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // std::cout << "enqueue ND Range kernel finished" << std::endl;

        start = std::chrono::steady_clock::now();
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
        end = std::chrono::steady_clock::now();
        stats.copy_data_to_host 
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    }
    catch (cl::Error err)
    {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
    }

    return stats;

}

void CLRaycaster::castRays(
    const std::vector<Vector3f >& origins,
    const std::vector<Vector3f >& directions,
    std::vector<Vector3f >& intersections,
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

void CLRaycaster::testKernel(
    const Vector3f& origin,
    const std::vector<Vector3f >& directions)
{
    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
    const float* direction_f = reinterpret_cast<const float*>(directions.data());

    try { 
        initOpenCLRayBuffer(3, directions.size()*3);
        copyRayDataToGPU(origin_f, 3, direction_f, directions.size()*3);

        const unsigned int N = directions.size();

        m_kernel_test.setArg(0, m_rayOriginBuffer);
        m_kernel_test.setArg(1, m_rayBuffer);
        m_kernel_test.setArg(2, N);

        cl::Event evt;

        m_queue.enqueueNDRangeKernel(
            m_kernel_test,
            cl::NullRange,
            cl::NDRange(1024),
            cl::NDRange(32),
            nullptr,
            &evt
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
void CLRaycaster::initOpenCL()
{
    // std::cout << "Get platforms" << std::endl;
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    // for (auto const& platform: platforms)
    // {
    //     std::cout << "Found platform: " 
    //         << platform.getInfo<CL_PLATFORM_NAME>().c_str() 
    //         << std::endl;
    // }
    // std::cout << std::endl;

    vector<cl::Device> consideredDevices;
    for (auto const& platform: platforms)
    {
        // std::cout << "Get devices of " << platform.getInfo<CL_PLATFORM_NAME>().c_str() << ": " << std::endl;
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
            // std::cout << "Found device: " << device.getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
            // std::cout << "Device work units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            // std::cout << "Device work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

            consideredDevices.push_back(device);
        }
    }
    // std::cout << std::endl;

    // preferably choose the first compatible device of type GPU
    bool deviceFound = false;
    for (auto const& device : consideredDevices)
    {
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

void CLRaycaster::getDeviceInformation()
{
    cl_int ret;

    // compute units
    m_mps = m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&ret);

    // max work item dimensions
    cl_uint max_work_item_dimensions 
        = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(&ret);

    // max work item sizes
    std::vector<size_t> max_work_item_sizes
        = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&ret);
    m_threads_per_block = max_work_item_sizes[0];

    // max work group size
    m_max_work_group_size 
        = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&ret);

    // memory
    m_device_global_memory
        = m_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&ret);
}

void CLRaycaster::initOpenCLTreeBuffer()
{

    // create buffers on the device
    m_bvhIndicesOrTriListsBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * BVHRaycaster::m_bvh.getIndexesOrTrilists().size()
    );
    m_bvhLimitsnBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * BVHRaycaster::m_bvh.getLimits().size()
    );
    m_bvhTriangleIntersectionDataBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * BVHRaycaster::m_bvh.getTrianglesIntersectionData().size()
    );
    m_bvhTriIdxListBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * BVHRaycaster::m_bvh.getTriIndexList().size()
    );

}

void CLRaycaster::initOpenCLRayBuffer(int num_origins, int num_rays)
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

void CLRaycaster::copyBVHToGPU()
{
    m_queue.enqueueWriteBuffer(
        m_bvhIndicesOrTriListsBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * BVHRaycaster::m_bvh.getIndexesOrTrilists().size(),
        BVHRaycaster::m_bvh.getIndexesOrTrilists().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhLimitsnBuffer,
        CL_TRUE,
        0,
        sizeof(float) * BVHRaycaster::m_bvh.getLimits().size(),
        BVHRaycaster::m_bvh.getLimits().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriangleIntersectionDataBuffer,
        CL_TRUE,
        0,
        sizeof(float) * BVHRaycaster::m_bvh.getTrianglesIntersectionData().size(),
        BVHRaycaster::m_bvh.getTrianglesIntersectionData().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriIdxListBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * BVHRaycaster::m_bvh.getTriIndexList().size(),
        BVHRaycaster::m_bvh.getTriIndexList().data()
    );

}

void CLRaycaster::createKernel()
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

void CLRaycaster::copyRayDataToGPU(
    const vector<float>& origins,
    const vector<float>& rays
)
{
    copyRayDataToGPU(origins.data(), origins.size(), rays.data(), rays.size());
}

void CLRaycaster::copyRayDataToGPU(
    const float* origin_buffer, size_t origin_buffer_size,
    const float* ray_buffer, size_t ray_buffer_size
)
{
    // std::cout << "Number of origins: " << origin_buffer_size/3 << endl;
    m_queue.enqueueWriteBuffer(m_rayOriginBuffer, CL_TRUE, 0, sizeof(float) * origin_buffer_size, origin_buffer);

    // std::cout << "Size of rays: " << ray_buffer_size/3 << endl;

    // std::cout << ray_buffer[297] << std::endl;
    m_queue.enqueueWriteBuffer(m_rayBuffer, CL_TRUE, 0, sizeof(float) * ray_buffer_size, ray_buffer);
}


} // namespace lvr2
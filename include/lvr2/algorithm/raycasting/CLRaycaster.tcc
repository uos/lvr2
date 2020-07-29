
#include <sstream>

namespace lvr2 {

constexpr char CAST_RAYS_BVH_PROGRAM[] =
    #include "opencl/cast_rays_bvh.cl"
;

template<typename IntT>
CLRaycaster<IntT>::CLRaycaster(const MeshBufferPtr mesh, 
    unsigned int stack_size)
:BVHRaycaster<IntT>(mesh, stack_size)
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

template<typename IntT>
bool CLRaycaster<IntT>::castRay(
    const Vector3f& origin,
    const Vector3f& direction,
    IntT& intersection)
{
    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
    const float* direction_f = reinterpret_cast<const float*>(&direction.coeffRef(0));

    ClTriangleIntersectionResult result;

    // #pragma omp critical
    // {
    try {
        
        // 1. INIT BUFFER
        // std::cout << "OpenCL init buffer" << std::endl;
        // input buffer
        cl::Buffer rayOriginBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3
        );
        cl::Buffer rayDirBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3
        );
        // output buffer
        cl::Buffer resultBuffer(
            m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            sizeof(ClTriangleIntersectionResult)
        );

        // 2. COPY BUFFER
        // std::cout << "Copy to GPU" << std::endl;
        m_queue.enqueueWriteBuffer(rayOriginBuffer, CL_TRUE, 0, sizeof(float) * 3, origin_f);
        m_queue.enqueueWriteBuffer(rayDirBuffer, CL_TRUE, 0, sizeof(float) * 3, direction_f);

        
        // 3. SET KERNEL ARGS
        // std::cout << "Set kernel args" << std::endl;
        m_kernel_one_one.setArg(0, rayOriginBuffer);
        m_kernel_one_one.setArg(1, rayDirBuffer);
        m_kernel_one_one.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_one_one.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_one_one.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_one_one.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_one_one.setArg(6, resultBuffer);

        // 4. LAUNCH KERNEL
        // std::cout << "Launch kernel" << std::endl;
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

        // 5. READ DATA
        // std::cout << "Data GPU -> CPU" << std::endl;
        m_queue.enqueueReadBuffer(
            resultBuffer,
            CL_TRUE,
            0,
            sizeof(ClTriangleIntersectionResult),
            &result
        );
        
        m_queue.finish();

        // std::cout << "hit dist: " << std::endl;
        // std::cout << result.pointHit.x << std::endl;

    } catch (cl::Error err) {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
        throw std::runtime_error(CLUtil::getErrorDescription(err.err()));
    }
    // }

    // convert
    if constexpr(IntT::template has<lvr2::intelem::Point>())
    {
        intersection.point.x() = result.pointHit.x;
        intersection.point.y() = result.pointHit.y;
        intersection.point.z() = result.pointHit.z;
    }

    if constexpr(IntT::template has<lvr2::intelem::Distance>())
    {
        intersection.dist = result.hitDist;
    }

    if constexpr(IntT::template has<lvr2::intelem::Normal>())
    {
        unsigned int v1id = m_faces[result.pBestTriId * 3 + 0];
        unsigned int v2id = m_faces[result.pBestTriId * 3 + 1];
        unsigned int v3id = m_faces[result.pBestTriId * 3 + 2];

        Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
        Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
        Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);

        intersection.normal = (v3 - v1).cross((v2 - v1));
        intersection.normal.normalize();
        if(direction.dot(intersection.normal) > 0.0)
        {
            intersection.normal = -intersection.normal;
        }
    }

    if constexpr(IntT::template has<lvr2::intelem::Face>())
    {
        intersection.face_id = result.pBestTriId;
    }

    if constexpr(IntT::template has<lvr2::intelem::Barycentrics>())
    {
        unsigned int v1id = m_faces[result.pBestTriId * 3 + 0];
        unsigned int v2id = m_faces[result.pBestTriId * 3 + 1];
        unsigned int v3id = m_faces[result.pBestTriId * 3 + 2];
        
        Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
        Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
        Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);
    
        Vector3f bary = barycentric(Vector3f(
                    result.pointHit.x, 
                    result.pointHit.y, 
                    result.pointHit.z), 
                v1, v2, v3);
        intersection.b_uv.x() = bary.x();
        intersection.b_uv.y() = bary.y();
    }

    if constexpr(IntT::template has<lvr2::intelem::Mesh>())
    {
        // TODO
        intersection.mesh_id = 0;
    }

    return result.hit;
}

template<typename IntT>
void CLRaycaster<IntT>::castRays(
    const Vector3f& origin,
    const std::vector<Vector3f>& directions,
    std::vector<IntT>& intersections,
    std::vector<uint8_t>& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size(), false);
    std::vector<ClTriangleIntersectionResult> result(directions.size());

    const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
    const float* directions_f = reinterpret_cast<const float*>(&directions[0]);

    try {
        
        // 1. INIT BUFFER
        // std::cout << "OpenCL init buffer" << std::endl;
        // input buffer
        cl::Buffer rayOriginBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3
        );
        cl::Buffer rayDirBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3 * directions.size()
        );
        // output buffer
        cl::Buffer resultBuffer(
            m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            sizeof(ClTriangleIntersectionResult) * directions.size()
        );

        // 2. COPY BUFFER
        // std::cout << "Copy to GPU" << std::endl;
        m_queue.enqueueWriteBuffer(rayOriginBuffer, CL_TRUE, 0, sizeof(float) * 3, origin_f);
        m_queue.enqueueWriteBuffer(rayDirBuffer, CL_TRUE, 0, sizeof(float) * 3 * directions.size(), directions_f);

        
        // 3. SET KERNEL ARGS
        // std::cout << "Set kernel args" << std::endl;
        m_kernel_one_multi.setArg(0, rayOriginBuffer);
        m_kernel_one_multi.setArg(1, rayDirBuffer);
        m_kernel_one_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_one_multi.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_one_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_one_multi.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_one_multi.setArg(6, resultBuffer);

        // 4. LAUNCH KERNEL
        // std::cout << "Launch kernel" << std::endl;
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

        // 5. READ DATA
        // std::cout << "Data GPU -> CPU" << std::endl;
        m_queue.enqueueReadBuffer(
            resultBuffer,
            CL_TRUE,
            0,
            sizeof(ClTriangleIntersectionResult) * directions.size(),
            &result[0]
        );
        
        m_queue.finish();
    } catch (cl::Error err) {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
        throw std::runtime_error(CLUtil::getErrorDescription(err.err()));
    }

    // CONVERT

    for(size_t i=0; i<intersections.size(); i++)
    {
        hits[i] = result[i].hit;

        if constexpr(IntT::template has<lvr2::intelem::Point>())
        {
            intersections[i].point.x() = result[i].pointHit.x;
            intersections[i].point.y() = result[i].pointHit.y;
            intersections[i].point.z() = result[i].pointHit.z;
        }

        if constexpr(IntT::template has<lvr2::intelem::Distance>())
        {
            intersections[i].dist = result[i].hitDist;
        }

        if constexpr(IntT::template has<lvr2::intelem::Normal>())
        {
            unsigned int v1id = m_faces[result[i].pBestTriId * 3 + 0];
            unsigned int v2id = m_faces[result[i].pBestTriId * 3 + 1];
            unsigned int v3id = m_faces[result[i].pBestTriId * 3 + 2];

            Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
            Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
            Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);

            Vector3f normal = (v3 - v1).cross((v2 - v1));
            normal.normalize();
            if(directions[i].dot(normal) > 0.0)
            {
                normal = -normal;
            }

            intersections[i].normal = normal;
        }

        if constexpr(IntT::template has<lvr2::intelem::Face>())
        {
            intersections[i].face_id = result[i].pBestTriId;
        }

        if constexpr(IntT::template has<lvr2::intelem::Barycentrics>())
        {
            unsigned int v1id = m_faces[result[i].pBestTriId * 3 + 0];
            unsigned int v2id = m_faces[result[i].pBestTriId * 3 + 1];
            unsigned int v3id = m_faces[result[i].pBestTriId * 3 + 2];
            
            Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
            Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
            Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);
        
            ;
            Vector3f bary = barycentric(
                Vector3f(result[i].pointHit.x, 
                         result[i].pointHit.y, 
                         result[i].pointHit.z), 
                v1, v2, v3);
            intersections[i].b_uv.x() = bary.x();
            intersections[i].b_uv.y() = bary.y();
        }

        if constexpr(IntT::template has<lvr2::intelem::Mesh>())
        {
            // TODO
            intersections[i].mesh_id = 0;
        }
    }
}


// template<typename IntT>
// void CLRaycaster<IntT>::castRays(
//     const Vector3f& origin,
//     const std::vector<std::vector<Vector3f> >& directions,
//     std::vector< std::vector<IntT> >& intersections,
//     std::vector< std::vector<uint8_t> >& hits)
// {
//     std::cout << "BLAAA" << std::endl;
//     intersections.resize(directions.size());
//     hits.resize(directions.size());
//     // std::vector<std::vector<ClTriangleIntersectionResult> > result(directions.size());

//     std::vector<const Vector3f*> directions_hack;

//     size_t ndirections = 0;

//     for (auto&& vec : directions)
//     {
//         // intersections.resize(vec.size());
//         // hits[i].resize(vec.size(), false);
//         directions_hack.push_back(&vec[0]);
//         ndirections += vec.size();
//     }

//     // for(size_t i=0; i<directions.size(); i++)
//     // {
//     //     intersections[i].resize(directions[i].size());
//     //     hits[i].resize(directions[i].size(), false);
//     //     directions_hack[i] = &(directions[i][0]);
//     //     // result[i].resize(directions[i].size());
//     //     ndirections += directions[i].size();
//     // }

//     std::cout << "ndirs " << ndirections << std::endl;

//     std::vector<ClTriangleIntersectionResult> result(ndirections);

//     const float* origin_f = reinterpret_cast<const float*>(&origin.coeffRef(0));
//     const float* directions_f = reinterpret_cast<const float*>(&directions_hack[0]);

//     try {
        
//         // 1. INIT BUFFER
//         std::cout << "OpenCL init buffer" << std::endl;
//         // input buffer
//         cl::Buffer rayOriginBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
//             sizeof(float) * 3
//         );
//         cl::Buffer rayDirBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
//             sizeof(float) * 3 * ndirections
//         );
//         // output buffer
//         cl::Buffer resultBuffer(
//             m_context,
//             CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
//             sizeof(ClTriangleIntersectionResult) * ndirections
//         );

//         // 2. COPY BUFFER
//         std::cout << "Copy to GPU" << std::endl;
//         m_queue.enqueueWriteBuffer(rayOriginBuffer, CL_TRUE, 0, sizeof(float) * 3, origin_f);
//         m_queue.enqueueWriteBuffer(rayDirBuffer, CL_TRUE, 0, sizeof(float) * 3 * ndirections, directions_f);

        
//         // 3. SET KERNEL ARGS
//         // std::cout << "Set kernel args" << std::endl;
//         m_kernel_one_multi.setArg(0, rayOriginBuffer);
//         m_kernel_one_multi.setArg(1, rayDirBuffer);
//         m_kernel_one_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
//         m_kernel_one_multi.setArg(3, m_bvhLimitsnBuffer);
//         m_kernel_one_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
//         m_kernel_one_multi.setArg(5, m_bvhTriIdxListBuffer);
//         m_kernel_one_multi.setArg(6, resultBuffer);

//         // 4. LAUNCH KERNEL
//         std::cout << "Launch kernel" << std::endl;
//         cl::Event evt;
//         m_queue.enqueueNDRangeKernel(
//             m_kernel_one_multi,
//             cl::NullRange,
//             cl::NDRange(ndirections),
//             cl::NullRange,
//             nullptr,
//             &evt
//         );
//         m_queue.finish();

//         // 5. READ DATA
//         std::cout << "Data GPU -> CPU" << std::endl;
//         m_queue.enqueueReadBuffer(
//             resultBuffer,
//             CL_TRUE,
//             0,
//             sizeof(ClTriangleIntersectionResult) * ndirections,
//             &result[0]
//         );
        
//         m_queue.finish();
//     } catch (cl::Error err) {
//         std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
//         std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
//         throw std::runtime_error(CLUtil::getErrorDescription(err.err()));
//     }

//     // CONVERT
//     size_t elem_id = 0;
//     for(size_t i=0; i<intersections.size(); i++)
//     {
//         for(size_t j=0; j<intersections[i].size(); j++, elem_id++)
//         {
//             std::cout << result[elem_id].hitDist << std::endl;
//             // hits[i][j] = result[elem_id].hit;

//             // std::cout << directions[i][j].transpose() << std::endl;

//             // if constexpr(IntT::template has<lvr2::intelem::Point>())
//             // {
//             //     intersections[i][j].point.x() = result[elem_id].pointHit.x;
//             //     intersections[i][j].point.y() = result[elem_id].pointHit.y;
//             //     intersections[i][j].point.z() = result[elem_id].pointHit.z;
//             // }

//             // if constexpr(IntT::template has<lvr2::intelem::Distance>())
//             // {
//             //     intersections[i][j].dist = result[i][j].hitDist;
//             // }

//             // if constexpr(IntT::template has<lvr2::intelem::Normal>())
//             // {
//             //     unsigned int v1id = m_faces[result[elem_id].pBestTriId * 3 + 0];
//             //     unsigned int v2id = m_faces[result[elem_id].pBestTriId * 3 + 1];
//             //     unsigned int v3id = m_faces[result[elem_id].pBestTriId * 3 + 2];

//             //     Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
//             //     Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
//             //     Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);

//             //     Vector3f normal = (v3 - v1).cross((v2 - v1));
//             //     normal.normalize();
//             //     if(directions[i][j].dot(normal) > 0.0)
//             //     {
//             //         normal = -normal;
//             //     }

//             //     intersections[i][j].normal = normal;
//             // }

//             // if constexpr(IntT::template has<lvr2::intelem::Face>())
//             // {
//             //     intersections[i][j].face_id = result[elem_id].pBestTriId;
//             // }

//             // if constexpr(IntT::template has<lvr2::intelem::Barycentrics>())
//             // {
//             //     unsigned int v1id = m_faces[result[elem_id].pBestTriId * 3 + 0];
//             //     unsigned int v2id = m_faces[result[elem_id].pBestTriId * 3 + 1];
//             //     unsigned int v3id = m_faces[result[elem_id].pBestTriId * 3 + 2];
                
//             //     Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
//             //     Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
//             //     Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);
            
//             //     ;
//             //     Vector3f bary = barycentric(
//             //         Vector3f(result[elem_id].pointHit.x, 
//             //                 result[elem_id].pointHit.y, 
//             //                 result[elem_id].pointHit.z), 
//             //         v1, v2, v3);
//             //     intersections[i][j].b_uv.x() = bary.x();
//             //     intersections[i][j].b_uv.y() = bary.y();
//             // }

//             // if constexpr(IntT::template has<lvr2::intelem::Mesh>())
//             // {
//             //     // TODO
//             //     intersections[i][j].mesh_id = 0;
//             // }
//         }
//     }
// }

template<typename IntT>
void CLRaycaster<IntT>::castRays(
    const std::vector<Vector3f>& origins,
    const std::vector<Vector3f>& directions,
    std::vector<IntT>& intersections,
    std::vector<uint8_t>& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size(), false);
    std::vector<ClTriangleIntersectionResult> result(directions.size());

    const float* origins_f = reinterpret_cast<const float*>(&origins[0]);
    const float* directions_f = reinterpret_cast<const float*>(&directions[0]);

    try {
        
        // 1. INIT BUFFER
        std::cout << "OpenCL init buffer" << std::endl;
        // input buffer
        cl::Buffer rayOriginBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3 * origins.size()
        );
        cl::Buffer rayDirBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            sizeof(float) * 3 * directions.size()
        );
        // output buffer
        cl::Buffer resultBuffer(
            m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            sizeof(ClTriangleIntersectionResult) * directions.size()
        );

        // 2. COPY BUFFER
        // std::cout << "Copy to GPU" << std::endl;
        m_queue.enqueueWriteBuffer(rayOriginBuffer, CL_TRUE, 0, sizeof(float) * 3 * origins.size(), origins_f);
        m_queue.enqueueWriteBuffer(rayDirBuffer, CL_TRUE, 0, sizeof(float) * 3 * directions.size(), directions_f);

        m_queue.finish();
        // 3. SET KERNEL ARGS
        // std::cout << "Set kernel args" << std::endl;
        m_kernel_multi_multi.setArg(0, rayOriginBuffer);
        m_kernel_multi_multi.setArg(1, rayDirBuffer);
        m_kernel_multi_multi.setArg(2, m_bvhIndicesOrTriListsBuffer);
        m_kernel_multi_multi.setArg(3, m_bvhLimitsnBuffer);
        m_kernel_multi_multi.setArg(4, m_bvhTriangleIntersectionDataBuffer);
        m_kernel_multi_multi.setArg(5, m_bvhTriIdxListBuffer);
        m_kernel_multi_multi.setArg(6, resultBuffer);

        // 4. LAUNCH KERNEL
        // std::cout << "Launch kernel" << std::endl;
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

        // 5. READ DATA
        // std::cout << "Data GPU -> CPU" << std::endl;
        m_queue.enqueueReadBuffer(
            resultBuffer,
            CL_TRUE,
            0,
            sizeof(ClTriangleIntersectionResult) * directions.size(),
            &result[0]
        );
        
        m_queue.finish();
    } catch (cl::Error err) {
        std::cerr << err.what() << ": " << CLUtil::getErrorString(err.err()) << std::endl;
        std::cout << "(" << CLUtil::getErrorDescription(err.err()) << ")" << std::endl;
        throw std::runtime_error(CLUtil::getErrorDescription(err.err()));
    }

    // CONVERT

    for(size_t i=0; i<intersections.size(); i++)
    {
        hits[i] = result[i].hit;

        if constexpr(IntT::template has<lvr2::intelem::Point>())
        {
            intersections[i].point.x() = result[i].pointHit.x;
            intersections[i].point.y() = result[i].pointHit.y;
            intersections[i].point.z() = result[i].pointHit.z;
        }

        if constexpr(IntT::template has<lvr2::intelem::Distance>())
        {
            intersections[i].dist = result[i].hitDist;
        }

        if constexpr(IntT::template has<lvr2::intelem::Normal>())
        {
            unsigned int v1id = m_faces[result[i].pBestTriId * 3 + 0];
            unsigned int v2id = m_faces[result[i].pBestTriId * 3 + 1];
            unsigned int v3id = m_faces[result[i].pBestTriId * 3 + 2];

            Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
            Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
            Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);

            Vector3f normal = (v3 - v1).cross((v2 - v1));
            normal.normalize();
            if(directions[i].dot(normal) > 0.0)
            {
                normal = -normal;
            }

            intersections[i].normal = normal;
        }

        if constexpr(IntT::template has<lvr2::intelem::Face>())
        {
            intersections[i].face_id = result[i].pBestTriId;
        }

        if constexpr(IntT::template has<lvr2::intelem::Barycentrics>())
        {
            unsigned int v1id = m_faces[result[i].pBestTriId * 3 + 0];
            unsigned int v2id = m_faces[result[i].pBestTriId * 3 + 1];
            unsigned int v3id = m_faces[result[i].pBestTriId * 3 + 2];
            
            Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
            Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
            Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);
        
            ;
            Vector3f bary = barycentric(
                Vector3f(result[i].pointHit.x, 
                         result[i].pointHit.y, 
                         result[i].pointHit.z), 
                v1, v2, v3);
            intersections[i].b_uv.x() = bary.x();
            intersections[i].b_uv.y() = bary.y();
        }

        if constexpr(IntT::template has<lvr2::intelem::Mesh>())
        {
            // TODO
            intersections[i].mesh_id = 0;
        }
    }
}

// PRIVATE FUNCTIONS
template<typename IntT>
void CLRaycaster<IntT>::initOpenCL()
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
        throw std::runtime_error("No device with compatible OpenCL version found (minimum 2.0)");
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
        throw err;
    }

    m_queue = cl::CommandQueue(m_context, m_device, 0);
}

template<typename IntT>
void CLRaycaster<IntT>::getDeviceInformation()
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

template<typename IntT>
void CLRaycaster<IntT>::initOpenCLTreeBuffer()
{
    // create buffers on the device
    m_bvhIndicesOrTriListsBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * m_bvh.getIndexesOrTrilists().size()
    );
    m_bvhLimitsnBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * m_bvh.getLimits().size()
    );
    m_bvhTriangleIntersectionDataBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * m_bvh.getTrianglesIntersectionData().size()
    );
    m_bvhTriIdxListBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(uint32_t) * m_bvh.getTriIndexList().size()
    );
}

template<typename IntT>
void CLRaycaster<IntT>::initOpenCLBuffer(
    size_t num_origins,
    size_t num_dirs)
{

    // input buffer
    m_rayOriginBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * num_origins * 3
    );

    // std::cout << "origin buffer bytes: " << sizeof(float) * num_origins * 3 << std::endl;

    m_rayBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        sizeof(float) * num_dirs * 3
    );

    // std::cout << "dir buffer bytes: " << sizeof(float) * num_dirs * 3 << std::endl;

    // output buffer
    m_resultBuffer = cl::Buffer(
        m_context,
        CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
        sizeof(ClTriangleIntersectionResult) * num_dirs
    );
    
}

template<typename IntT>
void CLRaycaster<IntT>::copyBVHToGPU()
{
    m_queue.enqueueWriteBuffer(
        m_bvhIndicesOrTriListsBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * m_bvh.getIndexesOrTrilists().size(),
        m_bvh.getIndexesOrTrilists().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhLimitsnBuffer,
        CL_TRUE,
        0,
        sizeof(float) * m_bvh.getLimits().size(),
        m_bvh.getLimits().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriangleIntersectionDataBuffer,
        CL_TRUE,
        0,
        sizeof(float) * m_bvh.getTrianglesIntersectionData().size(),
        m_bvh.getTrianglesIntersectionData().data()
    );
    m_queue.enqueueWriteBuffer(
        m_bvhTriIdxListBuffer,
        CL_TRUE,
        0,
        sizeof(uint32_t) * m_bvh.getTriIndexList().size(),
        m_bvh.getTriIndexList().data()
    );

}

template<typename IntT>
void CLRaycaster<IntT>::createKernel()
{
    // one origin one ray
    m_kernel_one_one = cl::Kernel(m_program, "cast_rays_one_one");

    // one origin multiple rays
    m_kernel_one_multi = cl::Kernel(m_program, "cast_rays_one_multi");

    // multiple origins multiple rays
    m_kernel_multi_multi = cl::Kernel(m_program, "cast_rays_multi_multi");
}

} // namespace lvr2
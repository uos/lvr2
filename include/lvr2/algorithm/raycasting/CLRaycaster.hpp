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

/*
 * CLRaycaster.hpp
 *
 *  @date 25.01.2020
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 *  @author Alexander Mock <amock@uos.de>
 */

#ifndef LVR2_ALGORITHM_RAYCASTING_CLRAYCASTER
#define LVR2_ALGORITHM_RAYCASTING_CLRAYCASTER

#include <chrono>
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/algorithm/raycasting/BVHRaycaster.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120 // Need to set to 120 on CUDA 8
#define CL_HPP_TARGET_OPENCL_VERSION 120 // Need to set to 120 on CUDA 8

// hard cl2 depencency
#include <CL/cl2.hpp>

// if your code can handle cl.hpp and cl2.hpp use:
// #if defined LVR2_USE_OPENCL_NEW_API
//     #include <CL/cl2.hpp>
// #else
//     #include <CL/cl.hpp>
// #endif

#include "lvr2/util/CLUtil.hpp"
#include "Intersection.hpp"



namespace lvr2
{

/**
 *  @brief CLRaycaster: GPU OpenCL version of BVH Raycasting
 */
template<typename IntT>
class CLRaycaster : public BVHRaycaster<IntT> {
public:

    /**
     * @brief Constructor: Generate BVH tree on mesh, loads CL kernels
     */
    CLRaycaster(const MeshBufferPtr mesh,
                unsigned int stack_size = 32);

    /// Overload functions ///
    /**
     * @brief Cast a single ray onto the mesh. Hint: Better not use it on GPU.
     * 
     * @param[in] origin Ray origin 
     * @param[in] direction Ray direction
     * @param[out] intersection User defined intersection output 
     * @return true  Intersection found
     * @return false  Not intersection found
     */
    bool castRay(
        const Vector3f& origin,
        const Vector3f& direction,
        IntT& intersection);

    using BVHRaycaster<IntT>::castRays;

    /**
     * @brief Cast a ray from single origin 
     *        with multiple directions onto the mesh
     * 
     * @param[in] origin Origin of the ray
     * @param[in] directions Directions of the ray
     * @param[out] intersections User defined intersections output
     * @param[out] hits Intersection found or not
     */
    void castRays(
        const Vector3f& origin,
        const std::vector<Vector3f>& directions,
        std::vector<IntT>& intersections,
        std::vector<uint8_t>& hits) override;

    /**
     * @brief Cast from multiple ray origin/direction 
     *        pairs onto the mesh
     * 
     * @param[in] origin Origin of the ray
     * @param[in] directions Directions of the ray
     * @param[out] intersections User defined intersections output
     * @param[out] hits Intersection found or not
     */
    void castRays(
        const std::vector<Vector3f>& origins,
        const std::vector<Vector3f>& directions,
        std::vector<IntT>& intersections,
        std::vector<uint8_t>& hits) override;

    struct ClTriangleIntersectionResult {
        cl_uchar hit = 0;
        cl_uint pBestTriId;
        cl_float3 pointHit;
        cl_float hitDist;
    };

protected:
    using BVHRaycaster<IntT>::barycentric;
    using BVHRaycaster<IntT>::m_bvh;
    using BVHRaycaster<IntT>::m_vertices;
    using BVHRaycaster<IntT>::m_faces;

private:
    /**
     * @brief Initializes OpenCL related stuff
     */
    void initOpenCL();

    /**
     * @brief TODO
     */
    void getDeviceInformation();

    /**
     * @brief TODO docu
     */
    void initOpenCLTreeBuffer();

    /**
     * @brief TODO
     */
    // void initOpenCLRayBuffer(
    //     int num_origins,
    //     int num_rays);

    void initOpenCLBuffer(
        size_t num_origins,
        size_t num_dirs
    );

    /**
     * @brief TODO
     */
    void copyBVHToGPU();

    /**
     * @brief TODO
     */
    void createKernel();

    // /**
    //  * @brief TODO
    //  */
    // void copyRayDataToGPU(
    //     const vector<float>& origins,
    //     const vector<float>& rays
    // );

    // /**
    //  * @brief TODO
    //  */
    // void copyRayDataToGPU(
    //     const float* origin_buffer, size_t origin_buffer_size,
    //     const float* ray_buffer, size_t ray_buffer_size
    // );

    // Member vars

    // OpenCL Device information
    cl_uint m_mps;
    cl_uint m_threads_per_block;
    size_t m_max_work_group_size;
    size_t m_warp_size;
    cl_ulong m_device_global_memory;
    
    // OpenCL variables
    cl::Platform m_platform;
    cl::Device m_device;
    cl::Context m_context;
    cl::Program m_program;
    cl::CommandQueue m_queue;
    cl::Kernel m_kernel_one_one;
    cl::Kernel m_kernel_one_multi;
    cl::Kernel m_kernel_multi_multi;

    /// BUFFER ///
    // buffer bvh tree
    cl::Buffer m_bvhIndicesOrTriListsBuffer;
    cl::Buffer m_bvhLimitsnBuffer;
    cl::Buffer m_bvhTriangleIntersectionDataBuffer;
    cl::Buffer m_bvhTriIdxListBuffer;

    // buffer rays
    cl::Buffer m_rayBuffer;
    cl::Buffer m_rayOriginBuffer;

    // buffer results
    cl::Buffer m_resultBuffer;
    // cl::Buffer m_resultHitsBuffer;

};

} // namespace lvr2

#include "CLRaycaster.tcc"

#endif // LVR2_ALGORITHM_RAYCASTING_CLRAYCASTER
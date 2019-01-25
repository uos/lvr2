#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/BVH.hpp>

#include <lvr2/algorithm/raycasting/BVHRaycaster.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120 // Need to set to 120 on CUDA 8
#define CL_HPP_TARGET_OPENCL_VERSION 120 // Need to set to 120 on CUDA 8
#include <CL/cl2.hpp>
#include <lvr2/util/CLUtil.hpp>

const char *CAST_RAYS_BVH_PROGRAM =
    #include "opencl/cast_rays_bvh.cl"
;

namespace lvr2
{

/**
 *  @brief CLRaycaster: GPU OpenCL version of BVH Raycasting: WIP
 */
template <typename BaseVecT>
class CLRaycaster : public BVHRaycaster<BaseVecT> {
public:

    /**
     * @brief Constructor: Generate BVH tree on mesh, loads CL kernels
     */
    CLRaycaster(const MeshBufferPtr mesh);

    /// Overload functions ///

    Point<BaseVecT> castRay(
        const Point<BaseVecT>& origin,
        const Vector<BaseVecT>& direction
    );

    std::vector<Point<BaseVecT> > castRays(
        const Point<BaseVecT>& origin,
        const std::vector<Vector<BaseVecT> >& directions
    );

    std::vector<Point<BaseVecT> > castRays(
        const std::vector<Point<BaseVecT> >& origins,
        const std::vector<Vector<BaseVecT> >& directions
    );

private:
    /**
     * @brief Initializes OpenCL related stuff
     */
    void initOpenCL();

    /**
     * @brief TODO docu
     */
    void initOpenCLTreeBuffer();

    /**
     * @brief TODO
     */
    void initOpenCLRayBuffer(int num_origins, int num_rays);

    /**
     * @brief TODO
     */
    void copyBVHToGPU();

    /**
     * @brief TODO
     */
    void createKernel();

    /**
     * @brief TODO
     */
    void copyRayDataToGPU(
        const vector<float>& origins,
        const vector<float>& rays
    );

    /**
     * @brief TODO
     */
    void copyRayDataToGPU(
        const float* origins, size_t num_origins,
        const float* rays, size_t num_rays
    );

    // Member vars
    // OpenCL variables
    cl::Platform m_platform;
    cl::Device m_device;
    cl::Context m_context;
    cl::Program m_program;
    cl::CommandQueue m_queue;
    cl::Kernel m_kernel_one_one;
    cl::Kernel m_kernel_one_multi;
    cl::Kernel m_kernel_multi_multi;
    cl::Kernel m_faster_kernel;


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
    cl::Buffer m_resultHitsBuffer;

};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/CLRaycaster.tcc>
#ifndef LVR2_RECONSTRUCTION_CUDAKSEARCHSURFACE_HPP_
#define LVR2_RECONSTRUCTION_CUDAKSEARCHSURFACE_HPP_

#include "cuda/LBVHIndex.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"

#include "PointsetSurface.hpp"

namespace lvr2
{

template<typename BaseVecT>
class CudaKSearchSurface : public PointsetSurface<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;
public:
    CudaKSearchSurface(
        PointBufferPtr pbuffer,
        size_t k
    );

    CudaKSearchSurface();

    virtual ~CudaKSearchSurface() {};

    // This function is not suited for GPU
    virtual std::pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
        distance(BaseVecT v) const;

    virtual void calculateSurfaceNormals();

protected:

    lbvh::LBVHIndex m_tree;
};

} // namespace lvr2

#include "CudaKSearchSurface.tcc"

#endif // LVR2_RECONSTRUCTION_CUDAKSEARCHSURFACE_HPP_
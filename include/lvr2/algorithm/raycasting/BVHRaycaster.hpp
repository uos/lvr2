#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/BVH.hpp>
#include <lvr2/algorithm/raycasting/RaycasterBase.hpp>

namespace lvr2
{


/**
 *  @brief BVHRaycaster: CPU version of BVH Raycasting: WIP
 */
template <typename BaseVecT>
class BVHRaycaster : public RaycasterBase<BaseVecT> {
public:

    /**
     * @brief Constructor: Stores mesh as member
     */
    BVHRaycaster(const MeshBufferPtr mesh);

    Point<BaseVecT> castRay(
        const Point<BaseVecT>& origin,
        const Vector<BaseVecT>& direction
    ) const;

    std::vector<Point<BaseVecT> > castRays(
        const Point<BaseVecT>& origin,
        const std::vector<Vector<BaseVecT> >& directions
    ) const;

    std::vector<Point<BaseVecT> > castRays(
        const std::vector<Point<BaseVecT> >& origins,
        const std::vector<Vector<BaseVecT> >& directions
    ) const;

private:
    BVHTree<BaseVecT> m_bvh;
};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/BVHRaycaster.tcc>
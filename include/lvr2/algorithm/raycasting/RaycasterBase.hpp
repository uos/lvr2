#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>

namespace lvr2
{

/**
 * @brief RaycasterBase interface
 */
template <typename BaseVecT>
class RaycasterBase {
public:

    /**
     * @brief Constructor: Stores mesh as member
     */
    RaycasterBase(const MeshBufferPtr mesh);

    virtual Point<BaseVecT> castRay(
        const Point<BaseVecT>& origin,
        const Vector<BaseVecT>& direction
    ) const = 0;

    virtual std::vector<Point<BaseVecT> > castRays(
        const Point<BaseVecT>& origin,
        const std::vector<Vector<BaseVecT> >& directions
    ) const = 0;

    virtual std::vector<Point<BaseVecT> > castRays(
        const std::vector<Point<BaseVecT> >& origins,
        const std::vector<Vector<BaseVecT> >& directions
    ) const = 0;

private:
    const MeshBufferPtr m_mesh;
};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/RaycasterBase.tcc>
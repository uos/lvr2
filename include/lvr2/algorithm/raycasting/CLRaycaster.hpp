#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/BVH.hpp>

#include <lvr2/algorithm/raycasting/BVHRaycaster.hpp>

namespace lvr2
{

template <typename BaseVecT>
class CLRaycaster : BVHRaycaster<BaseVecT> {
public:

    /**
     * @brief Constructor: Generate BVH tree on mesh, loads
     */
    CLRaycaster(const MeshBufferPtr mesh);

};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/CLRaycaster.tcc>
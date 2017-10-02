/*
 * PointsetGrid.hpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#ifndef _LVR2_RECONSTRUCTION_POINTSETGRID_H_
#define _LVR2_RECONSTRUCTION_POINTSETGRID_H_

#include "HashGrid.hpp"

#include "PointsetSurface.hpp"
#include <lvr2/geometry/BoundingBox.hpp>

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
class PointsetGrid: public HashGrid<BaseVecT, BoxT>
{
public:
    PointsetGrid(
        float cellSize,
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool isVoxelsize = true,
        bool extrude = true
    );

    virtual ~PointsetGrid() {}

    void calcDistanceValues();

private:

    /**
     * @brief Rounds the given value to the neares integer value
     */
    inline int calcIndex(float f)
    {
        return f < 0 ? f - .5 : f + .5;
    }

    PointsetSurfacePtr<BaseVecT> m_surface;
};

} // namespace lvr2

#include <lvr2/reconstruction/PointsetGrid.tcc>


#endif // _LVR2_RECONSTRUCTION_POINTSETGRID_H_
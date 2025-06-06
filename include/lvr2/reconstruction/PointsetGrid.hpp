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
 * PointsetGrid.hpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#ifndef _LVR2_RECONSTRUCTION_POINTSETGRID_H_
#define _LVR2_RECONSTRUCTION_POINTSETGRID_H_

#include "HashGrid.hpp"

#include "PointsetSurface.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
class PointsetGrid: public HashGrid<BaseVecT, BoxT>
{
public:
    /**
     * @brief Construct a new Pointset Grid object
     * 
     * @param resolution if isVoxelsize: voxel size. if not: number of voxels on the longest size of bb.
     * @param surface the surface to be used for the grid
     * @param bb the bounding box of the grid
     * @param isVoxelsize see resolution description
     * @param extrude add cells around the existing ones
     */
    PointsetGrid(
        float resolution,
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool isVoxelsize = true,
        bool extrude = true
    );

    virtual ~PointsetGrid() {}

    void calcDistanceValues();

private:

    PointsetSurfacePtr<BaseVecT> m_surface;
};

} // namespace lvr2

#include "lvr2/reconstruction/PointsetGrid.tcc"


#endif // _LVR2_RECONSTRUCTION_POINTSETGRID_H_

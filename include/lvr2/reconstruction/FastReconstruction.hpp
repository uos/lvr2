/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * FastReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_
#define _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/BoundingBox.hpp>

#include "PointsetMeshGenerator.hpp"
#include "LocalApproximation.hpp"
#include "FastBox.hpp"
#include "./SharpBox.hpp"
#include "BilinearFastBox.hpp"
#include "QueryPoint.hpp"
#include "PointsetSurface.hpp"
#include "HashGrid.hpp"


#include <unordered_map>
#include <memory>

using std::shared_ptr;
using std::unordered_map;

namespace lvr2
{

template<typename BaseVecT>
class FastReconstructionBase
{
public:
    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<BaseVecT> &mesh) = 0;

    virtual void getMesh(
        BaseMesh<BaseVecT>& mesh,
        BoundingBox<BaseVecT>& bb,
        vector<unsigned int>& duplicates,
        float comparePrecision
    ) = 0;
};

/**
 * @brief A surface reconstruction object that implements the standard
 *        marching cubes algorithm using a hashed grid structure for
 *        parallel computation.
 */
template<typename BaseVecT, typename BoxT>
class FastReconstruction : public FastReconstructionBase<BaseVecT>
{
public:

    /**
     * @brief Constructor.
     *
     * @param grid  A HashGrid instance on which the reconstruction is performed.
     */
    FastReconstruction(shared_ptr<HashGrid<BaseVecT, BoxT>> grid);


    /**
     * @brief Destructor.
     */
    virtual ~FastReconstruction() {};

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<BaseVecT> &mesh);

    virtual void getMesh(
        BaseMesh<BaseVecT>& mesh,
        BoundingBox<BaseVecT>& bb,
        vector<unsigned int>& duplicates,
        float comparePrecision
    );

private:

    shared_ptr<HashGrid<BaseVecT, BoxT>> m_grid;
};


} // namespace lvr2


#include <lvr2/reconstruction/FastReconstruction.tcc>

#endif /* _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_ */

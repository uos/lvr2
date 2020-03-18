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
 * FastReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_
#define _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

//#include "PointsetMeshGenerator.hpp"
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

    virtual ~FastReconstructionBase() = default;
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


#include "lvr2/reconstruction/FastReconstruction.tcc"

#endif /* _LVR2_RECONSTRUCTION_FASTRECONSTRUCTION_H_ */

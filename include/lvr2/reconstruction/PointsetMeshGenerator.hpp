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
 * Reconstructor.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_POINTSETMESHGENERATOR_H_
#define _LVR2_RECONSTRUCTION_POINTSETMESHGENERATOR_H_

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/reconstruction/MeshGenerator.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"

namespace lvr2
{

/**
 * @brief Interface class for surface reconstruction algorithms
 *        that generate triangle meshes from point set surfaces.
 */
template<typename BaseVecT>
class PointsetMeshGenerator : public MeshGenerator<BaseVecT>
{
public:

    /**
     * @brief Constructs a Reconstructor object using the given point
     *        set surface
     */
    PointsetMeshGenerator(PointsetSurfacePtr<BaseVecT> surface) : m_surface(surface) {}

    /**
     * @brief Generates a triangle mesh representation of the current
     *        point set.
     *
     * @param mesh      A surface representation of the current point
     *                  set.
     */
    virtual void getMesh(BaseMesh<BaseVecT>& mesh) = 0;

protected:

    /// The point cloud manager that handles the loaded point cloud data.
    PointsetSurfacePtr<BaseVecT> m_surface;
};

} //namespace lvr2

#endif /* _LVR2_RECONSTRUCTION_POINTSETMESHGENERATOR_H_ */

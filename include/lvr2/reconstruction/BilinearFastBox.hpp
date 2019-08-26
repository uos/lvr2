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

/**
 * BilinearFastBox.hpp
 *
 *  @date 16.02.2012
 *  @author Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_
#define _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_

#include "lvr2/reconstruction/FastBox.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"

namespace lvr2
{

template<typename BaseVecT>
class BilinearFastBox : public FastBox<BaseVecT>
{
public:
    BilinearFastBox(BaseVecT center);
    virtual ~BilinearFastBox();

    /**
     * @brief Performs a local reconstruction according to the standard
     *        Marching Cubes table from Paul Bourke.
     *
     * @param mesh          The reconstructed mesh
     * @param query_points  A vector containing the query points of the
     *                      reconstruction grid
     * @param globalIndex   The index of the newest vertex in the mesh, i.e.
     *                      a newly generated vertex shout have the index
     *                      globalIndex + 1.
     */
    virtual void getSurface(
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>>& query_points,
        uint &globalIndex
    );
    virtual void getSurface(
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>>& query_points,
        uint& globalIndex,
        BoundingBox<BaseVecT>& bb,
        vector<unsigned int>& duplicates,
        float comparePrecision
    );

    void optimizePlanarFaces(BaseMesh<BaseVecT>& mesh, size_t kc);

    // the point set surface
    static PointsetSurfacePtr<BaseVecT> m_surface;


private:
    vector<FaceHandle> m_faces;
    int m_mcIndex;

};

template<typename BaseVecT>
struct BoxTraits<BilinearFastBox<BaseVecT>>
{
    static const string type;
};



} // namespace lvr2

#include "lvr2/reconstruction/BilinearFastBox.tcc"

#endif /* _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_ */

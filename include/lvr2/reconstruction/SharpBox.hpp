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
 * SharpBox.hpp
 *
 *  @date 06.02.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

#ifndef SHARPBOX_H_
#define SHARPBOX_H_


#include "FastBox.hpp"
#include "MCTable.hpp"
#include <float.h>
#include "ExtendedMCTable.hpp"
#include "PointsetSurface.hpp"

namespace lvr2
{

/**
 * @brief Used for extended marching cubes Reconstruction.
 */
template<typename BaseVecT>
class SharpBox : public FastBox<BaseVecT>
{
public:
    SharpBox(BaseVecT center);
    virtual ~SharpBox();

    /**
     * @brief Performs a local reconstruction w.r.t. to sharp features
     *
     * @param mesh          The reconstructed mesh
     * @param query_points  A vector containing the query points of the
     *                      reconstruction grid
     * @param globalIndex   The index of the newest vertex in the mesh, i.e.
     *                      a newly generated vertex shout have the index
     *                      globalIndex + 1.
     */
    virtual void getSurface(
            BaseMesh<BaseVecT> &mesh,
            vector<QueryPoint<BaseVecT> > &query_points,
            uint &globalIndex);

    virtual void getSurface(
            std::vector<float>& vBuffer,
            std::vector<unsigned int>& fBuffer,
            vector<QueryPoint<BaseVecT> > &query_points,
            uint &globalIndex){}

    virtual void getSurface(
            BaseMesh<BaseVecT> &mesh,
            vector<QueryPoint<BaseVecT> > &query_points,
            uint &globalIndex,
            BoundingBox<BaseVecT> &bb,
            vector<unsigned int>& duplicates,
            float comparePrecision
    ){}

    // Threshold angle for sharp feature detection
    static float m_theta_sharp;

    // Threshold angle for corner detection
    static float m_phi_corner;

    // Indicates if the Box contains a Sharp Feature
    // used for Edge Flipping
    bool m_containsSharpFeature;

    // Indicates if the Box contains a Sharp Corner
    // used for Edge Flipping
    bool m_containsSharpCorner;

    // The surface index of the Extended MC-Table
    // used for Edge Flipping
    uint m_extendedMCIndex;

    // the point set surface
    static PointsetSurfacePtr<BaseVecT> m_surface;

private:
    /**
     * @brief gets the normals for the given vertices
     *
     * @param vertex_positions    The vertices
     * @param vertex_normals    This array holds the normals of the given vertices
     *                             after calling the method.
     */
    void getNormals(BaseVecT vertex_positions[],
                    Normal<typename BaseVecT::CoordType> vertex_normals[]);

    void detectSharpFeatures(BaseVecT vertex_positions[],
                             Normal<typename BaseVecT::CoordType> vertex_normals[], uint index);


    typedef SharpBox<BaseVecT> BoxType;
};

template<typename BaseVecT>
struct BoxTraits<SharpBox<BaseVecT> >
{
    static const string type;
};


} /* namespace lvr */

#include "SharpBox.tcc"

#endif /* SHARPBOX_H_ */

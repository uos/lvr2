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
#include <float.h>
#include "ExtendedMCTable.hpp"

namespace lvr
{

/**
 * @brief Used for extended marching cubes Reconstruction.
 */
template<typename VertexT, typename NormalT>
class SharpBox : public FastBox<VertexT, NormalT>
{
public:
    SharpBox(VertexT);
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
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);

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
    static typename PointsetSurface<VertexT>::Ptr m_surface;

private:
    /**
     * @brief gets the normals for the given vertices
     *
     * @param vertex_positions	The vertices
     * @param vertex_normals	This array holds the normals of the given vertices
     * 							after calling the method.
     */
    void getNormals(VertexT vertex_positions[], NormalT vertex_normals[]);

    void detectSharpFeatures(VertexT vertex_positions[], NormalT vertex_normals[], uint index);


    typedef SharpBox<VertexT, NormalT> BoxType;
};

template<typename VertexT, typename NormalT>
struct BoxTraits<SharpBox<VertexT, NormalT> >
{
	static const string type;
};


} /* namespace lvr */

#include "SharpBox.tcc"

#endif /* SHARPBOX_H_ */

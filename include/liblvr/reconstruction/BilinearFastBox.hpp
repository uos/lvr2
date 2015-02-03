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
 * BilinearFastBox.hpp
 *
 *  @date 16.02.2012
 *  @author Thomas Wiemann
 */

#ifndef BILINEARFASTBOX_HPP_
#define BILINEARFASTBOX_HPP_

#include "FastBox.hpp"
#include "PointsetSurface.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include "geometry/HalfEdgeFace.hpp"

namespace lvr
{

template<typename VertexT, typename NormalT>
class BilinearFastBox : public FastBox<VertexT, NormalT>
{
public:
    BilinearFastBox(VertexT &center);
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
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);

    void optimizePlanarFaces(typename PointsetSurface<VertexT>::Ptr surface, size_t kc);

private:
    vector<HalfEdgeFace<VertexT, NormalT>* > m_faces;
    int                                      m_mcIndex;

};


} /* namespace lvr */
#include "BilinearFastBox.tcc"

#endif /* BILINEARFASTBOX_HPP_ */

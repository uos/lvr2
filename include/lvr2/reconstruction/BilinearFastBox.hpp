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

#ifndef _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_
#define _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_

#include <lvr2/reconstruction/FastBox.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>

namespace lvr2
{

template<typename BaseVecT>
class BilinearFastBox : public FastBox<BaseVecT>
{
public:
    BilinearFastBox(Vector<BaseVecT> center);
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

#include <lvr2/reconstruction/BilinearFastBox.tcc>

#endif /* _LVR2_RECONSTRUCTION_BILINEARFASTBOX_H_ */

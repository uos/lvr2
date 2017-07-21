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
 * ClusterAlgorithm.tcc
 *
 *  @date 20.07.2017
 *  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 *  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 */

#include <lvr2/geometry/Cluster.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

namespace lvr2
{

template<typename BaseVecT>
std::vector<BaseVecT> calculateContour(ClusterHandle clusterH, HalfEdgeMesh<BaseVecT>& mesh, ClusterSet<FaceHandle>& clusterSet)
{
    std::vector<BaseVecT> result;
    auto cluster = clusterSet.getCluster(clusterH);

    for (auto faceH: cluster.handles)
    {
        std::vector<EdgeHandle> contours = mesh.getContourEdgesOfFace(
            faceH,
            [&, this](auto neighbourFaceH)
            {
                return (clusterH != clusterSet.getClusterH(neighbourFaceH));
            }
        );

        for (EdgeHandle edgeH: contours)
        {
            // TODO get vertices and store in result
        }
    }

    return result;
}


} // namespace lvr2
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
* Texturizer.tcc
*
*  @date 17.06.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#include "ClusterAlgorithm.hpp"

namespace lvr2
{

template<typename BaseVecT>
TexturizerResult generateTextures(
    HalfEdgeMesh<BaseVecT>& mesh,
    ClusterSet<FaceHandle>& faceHandleClusterSet,
    PointsetSurfacePtr<BaseVecT> surface
)
{
    int numFacesThreshold = 5;

    for (auto clusterH: faceHandleClusterSet)
    {
        const Cluster<FaceHandle> cluster = faceHandleClusterSet.getCluster(clusterH);
        int numFacesInCluster = cluster.handles.size();

        // only create textures for clusters that are large enough
        if (numFacesInCluster >= numFacesThreshold)
        {
            // contour
            std::vector<BaseVecT> contour = calculateContour(clusterH, mesh, faceHandleClusterSet);

            // initial texture

            // zuordnen & speichern
        }
    }

    return TexturizerResult();
}

} // namespace lvr2
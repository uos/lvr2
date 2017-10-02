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
 * ClusterAlgorithms.tcc
 *
 * @date 24.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <lvr2/algorithm/Planar.hpp>

namespace lvr2
{

template<typename BaseVecT>
void removeDanglingCluster(BaseMesh<BaseVecT>& mesh, size_t sizeThreshold)
{
    // Do cluster growing without a predicate, so cluster will consist of connected faces
    auto clusterSet = clusterGrowing(mesh, [](auto referenceFaceH, auto currentFaceH)
    {
        return true;
    });

    // Remove all faces in too small clusters
    for (auto clusterH: clusterSet)
    {
        auto cluster = clusterSet.getCluster(clusterH);
        if (cluster.handles.size() < sizeThreshold)
        {
            for (auto faceH: cluster.handles)
            {
                mesh.removeFace(faceH);
            }
        }
    }
}

} // namespace lvr2
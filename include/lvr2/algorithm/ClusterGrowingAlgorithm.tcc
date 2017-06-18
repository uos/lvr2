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
 * ClusterGrowingAlgorithm.hpp
 *
 *  @date 17.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <vector>

using std::vector;

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

template<typename BaseVecT>
ClusterSet<FaceHandle> ClusterGrowingAlgorithm<BaseVecT>::apply(const BaseMesh <BaseVecT>& mesh)
{
    ClusterSet<FaceHandle> clusterSet;
    VectorMap<FaceHandle, bool> visited(mesh.numFaces(), false);

    // Iterate over all faces
    for (auto faceH: mesh.faces())
    {
        // Check if face is in a cluster (e.g. we have not visited it)
        if (!visited[faceH])
        {
            // We found a not visited face. Prepare things for growing.
            vector<FaceHandle> stack;
            stack.push_back(faceH);
            auto cluster = clusterSet.createCluster();
            auto normal = mesh.getFaceNormal(faceH);

            // Grow my cluster, groOW!
            while (!stack.empty())
            {
                auto currentFace = stack.back();
                stack.pop_back();

                // Check if the last faces from stack and starting face are in the "same" plane
                if (mesh.getFaceNormal(currentFace).dot(normal.asVector()) > m_almostOne)
                {
                    // The face is in the "same" plane as the starting face => add it to cluster and mark as visited
                    clusterSet.addToCluster(cluster, currentFace);
                    visited.insert(currentFace, true);

                    // Find all unvisited neighbours of the current face and them to the stack
                    for (auto neighbour: mesh.getNeighboursOfFace(currentFace))
                    {
                        if (!visited[neighbour])
                        {
                            stack.push_back(neighbour);
                        }
                    }
                }
            }
        }
    }

    return clusterSet;
}

} // namespace lvr2

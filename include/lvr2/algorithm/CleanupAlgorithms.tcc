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
 * CleanupAlgorithms.tcc
 */

namespace lvr2
{

template<typename BaseVecT>
void cleanContours(BaseMesh<BaseVecT>& mesh, int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        for (const auto fH: mesh.faces())
        {
            // For each face, we want to count the number of boundary edges
            // adjacent to that face. This can be a number between 0 and 3.
            int boundaryEdgeCount = 0;
            for (const auto eH: mesh.getEdgesOfFace(fH))
            {
                // For both (optional) faces of the edge...
                for (const auto neighborFaceH: mesh.getFacesOfEdge(eH))
                {
                    // ... we will count one up if there is no face on that
                    // side. Note that this correctly ignores our own face.
                    if (!neighborFaceH)
                    {
                        boundaryEdgeCount += 1;
                    }
                }
            }

            // Now, given the number of boundary edges, we decide what to do
            // with the face.
            if (boundaryEdgeCount >= 2)
            {
                mesh.removeFace(fH);
            }
            else if (boundaryEdgeCount == 1)
            {
                // TODO: remove if area is small
            }
        }
    }
}

} // namespace lvr2

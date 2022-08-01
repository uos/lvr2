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

/*
 * PointsetGrid.tcc
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
PointsetGrid<BaseVecT, BoxT>::PointsetGrid(
    float resolution,
    PointsetSurfacePtr<BaseVecT> surface,
    BoundingBox<BaseVecT> bb,
    bool isVoxelsize,
    bool extrude
) :
    HashGrid<BaseVecT, BoxT>(resolution, bb, isVoxelsize, extrude),
    m_surface(surface)
{
    // Get indexed point buffer pointer
    auto numPoint = m_surface->pointBuffer()->numPoints();

    cout << timestamp << "Creating grid" << endl;

    FloatChannel pts = *(m_surface->pointBuffer()->getFloatChannel("points"));

    std::unordered_set<Vector3i> requiredCells;

    #pragma omp parallel
    {
        std::unordered_set<Vector3i> localCells;
        Vector3i index;
        #pragma omp for schedule(dynamic,64) nowait
        for(size_t i = 0; i < numPoint; i++)
        {
            BaseVecT point = pts[i];
            if (bb.contains(point))
            {
                this->calcIndex(point, index);
                localCells.insert(index);
            }
        }

        if (extrude)
        {
            std::unordered_set<Vector3i> innerOnly = localCells;
            for (auto& index : innerOnly)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dz = -1; dz <= 1; dz++)
                        {
                            localCells.insert(index + Vector3i(dx, dy, dz));
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            requiredCells.insert(localCells.begin(), localCells.end());
        }
    }

    this->addLatticePoints(requiredCells);
}

template<typename BaseVecT, typename BoxT>
void PointsetGrid<BaseVecT, BoxT>::calcDistanceValues()
{
    // Status message output
    string comment = timestamp.getElapsedTime() + "Calculating distance values ";
    ProgressBar progress(this->m_queryPoints.size(), comment);

    Timestamp ts;

    // Calculate a distance value for each query point
#ifndef MSVC
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for( int i = 0; i < (int)this->m_queryPoints.size(); i++){
        float projectedDistance;
        float euklideanDistance;

        //cout << euklideanDistance << " " << projectedDistance << endl;

        std::tie(projectedDistance, euklideanDistance) =
            this->m_surface->distance(this->m_queryPoints[i].m_position);
        if (euklideanDistance > 1.7320 * this->m_voxelsize)
        {
            this->m_queryPoints[i].m_invalid = true;
        }
        this->m_queryPoints[i].m_distance = projectedDistance;
        ++progress;
    }
    cout << endl << timestamp << "Elapsed time: " << ts.getElapsedTimeInS() << endl;
    return;
}

} // namespace lvr2

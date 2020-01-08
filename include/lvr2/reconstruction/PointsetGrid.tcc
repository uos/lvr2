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
    float cellSize,
    PointsetSurfacePtr<BaseVecT> surface,
    BoundingBox<BaseVecT> bb,
    bool isVoxelsize,
    bool extrude
) :
    HashGrid<BaseVecT, BoxT>(cellSize, bb, isVoxelsize, extrude),
    m_surface(surface)
{
    auto v_min = this->m_boundingBox.getMin();
    auto v_max = this->m_boundingBox.getMax();

    // Get indexed point buffer pointer
    auto numPoint = m_surface->pointBuffer()->numPoints();

    cout << timestamp << "Creating grid" << endl;

    FloatChannel pts = *(m_surface->pointBuffer()->getFloatChannel("points"));

    // Iterator over all points, calc lattice indices and add lattice points to the grid
    for(size_t i = 0; i < numPoint; i++)
    {
        BaseVecT pt = pts[i];
        auto index = (pt - v_min) / this->m_voxelsize;
        this->addLatticePoint(calcIndex(index.x), calcIndex(index.y), calcIndex(index.z));
    }
}


template<typename BaseVecT, typename BoxT>
void PointsetGrid<BaseVecT, BoxT>::calcDistanceValues()
{
    // Status message output
    string comment = timestamp.getElapsedTime() + "Calculating distance values ";
    ProgressBar progress(this->m_queryPoints.size(), comment);

    Timestamp ts;

    // Calculate a distance value for each query point
    #pragma omp parallel for schedule(dynamic, 4)
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
    cout << endl;
    cout << timestamp << "Elapsed time: " << ts.getElapsedTimeInS() << endl;
}

} // namespace lvr2

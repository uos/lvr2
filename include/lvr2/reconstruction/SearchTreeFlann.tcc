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
 * SearchTreeFlann.tcc
 *
 *  Created on: Sep 22, 2015
 *      Author: twiemann
 */

#include <lvr2/reconstruction/SearchTreeFlann.hpp>

#include <lvr2/io/Timestamp.hpp>

#include <lvr2/util/Panic.hpp>

using std::make_unique;

namespace lvr2
{

template<typename BaseVecT>
SearchTreeFlann<BaseVecT>::SearchTreeFlann(PointBufferPtr buffer)
{
    auto n = buffer->numPoints();
    FloatChannelOptional pts_optional = buffer->getFloatChannel("points");
    FloatChannel pts_channel = *pts_optional;
    auto flannPoints = flann::Matrix<CoordT>(new CoordT[3 * n], n, 3);
    for(size_t i = 0; i < n; i++)
    {
        Vector<BaseVecT> p = pts_channel[i];
        flannPoints[i][0] = p.x;
        flannPoints[i][1] = p.y;
        flannPoints[i][2] = p.z;
    }

    m_tree = make_unique<flann::Index<flann::L2_Simple<float>>>(
        flannPoints,
        ::flann::KDTreeSingleIndexParams(10, false)
    );
    m_tree->buildIndex();
}


template<typename BaseVecT>
void SearchTreeFlann<BaseVecT>::kSearch(
    const Vector<BaseVecT>& qp,
    int k,
    vector<size_t>& indices,
    vector<CoordT>& distances
) const
{
    flann::Matrix<float> query_point(new float[3], 1, 3);
    query_point[0][0] = qp.x;
    query_point[0][1] = qp.y;
    query_point[0][2] = qp.z;

    vector<int> flann_indices(k);
    vector<CoordT> flann_distances(k);

    flann::Matrix<int> ind(flann_indices.data(), 1, k);
    flann::Matrix<CoordT> dist(flann_distances.data(), 1, k);

    m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());

    for (int i = 0; i < k; i++)
    {
        indices.push_back(static_cast<size_t>(flann_indices[i]));
        distances.push_back(CoordT(flann_distances[i]));
    }
}

template<typename BaseVecT>
void SearchTreeFlann<BaseVecT>::radiusSearch(
    const Vector<BaseVecT>& qp,
    CoordT r,
    vector<size_t>& indices
) const
{
    panic_unimplemented("radiusSearch() is not implemented for FLANN yet");
}

} // namespace lvr2

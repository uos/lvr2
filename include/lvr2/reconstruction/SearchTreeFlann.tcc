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

#include "lvr2/reconstruction/SearchTreeFlann.hpp"

#include "lvr2/io/Timestamp.hpp"

#include "lvr2/util/Panic.hpp"

#ifndef __APPLE__
#include <omp.h>
#endif

using std::make_unique;

namespace lvr2
{

template<typename BaseVecT>
SearchTreeFlann<BaseVecT>::SearchTreeFlann(PointBufferPtr buffer)
{
    auto n = buffer->numPoints();
    FloatChannelOptional pts_optional = buffer->getFloatChannel("points");
    FloatChannel pts_channel = *pts_optional;

    m_data = boost::shared_array<CoordT>(new CoordT[3 * n]);
    auto flannPoints = flann::Matrix<CoordT>(m_data.get(), n, 3);
    for(size_t i = 0; i < n; i++)
    {
        BaseVecT p = pts_channel[i];
        flannPoints[i][0] = p.x;
        flannPoints[i][1] = p.y;
        flannPoints[i][2] = p.z;
    }

    m_tree = make_unique<flann::Index<flann::L2_Simple<CoordT>>>(
                 flannPoints,
                 ::flann::KDTreeSingleIndexParams(10, false)
             );
    m_tree->buildIndex();
}


template<typename BaseVecT>
int SearchTreeFlann<BaseVecT>::kSearch(
    const BaseVecT& qp,
    int k,
    vector<size_t>& indices,
    vector<CoordT>& distances
) const
{
    CoordT point[3] = { qp.x, qp.y, qp.z };
    flann::Matrix<CoordT> query_point(point, 1, 3);

    indices.resize(k);
    distances.resize(k);

    flann::Matrix<size_t> ind(indices.data(), 1, k);
    flann::Matrix<CoordT> dist(distances.data(), 1, k);

    return m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());
}

template<typename BaseVecT>
void SearchTreeFlann<BaseVecT>::radiusSearch(
    const BaseVecT& qp,
    CoordT r,
    vector<size_t>& indices
) const
{
    panic_unimplemented("radiusSearch() is not implemented for FLANN yet");
}

template<typename BaseVecT>
void SearchTreeFlann<BaseVecT>::kSearchMany(
    const BaseVecT* query,
    int n,
    int k,
    size_t* indices,
    CoordT* distances
) const
{
    CoordT* queries = new CoordT[n * 3];
    flann::Matrix<CoordT> queries_mat(queries, n, 3);
    flann::Matrix<size_t> indices_mat(indices, n, 1);
    flann::Matrix<CoordT> distances_mat(distances, n, 1);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        queries_mat[i][0] = query[i].x;
        queries_mat[i][1] = query[i].y;
        queries_mat[i][2] = query[i].z;
    }

    flann::SearchParams params;
    #ifndef __APPLE__
    params.cores = omp_get_max_threads();
    #else
    params.cores = 4;
    #endif
    m_tree->knnSearch(queries_mat, indices_mat, distances_mat, 1, params);

    delete[] queries;
}


} // namespace lvr2

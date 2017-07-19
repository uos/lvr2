/*
 * SearchTreeFlann.tcc
 *
 *  Created on: Sep 22, 2015
 *      Author: twiemann
 */

#include "SearchTreeFlann.hpp"

#include <lvr/io/Timestamp.hpp>

#include <lvr2/util/Panic.hpp>

using std::make_unique;

namespace lvr2
{

template<typename BaseVecT>
SearchTreeFlann<BaseVecT>::SearchTreeFlann(PointBufferPtr<BaseVecT> buffer)
{
    auto n = buffer->getNumPoints();
    auto flannPoints = flann::Matrix<CoordT>(new CoordT[3 * n], n, 3);
    for(size_t i = 0; i < n; i++)
    {
        auto p = buffer->getPoint(i);
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
    const Point<BaseVecT>& qp,
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
    const Point<BaseVecT>& qp,
    CoordT r,
    vector<size_t>& indices
) const
{
    panic_unimplemented("radiusSearch() is not implemented for FLANN yet");
}

} // namespace lvr2

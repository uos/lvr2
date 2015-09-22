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

/**
 * SearchTreeNanoflann.cpp
 *
 *  @date 29.08.2012
 *  @author Thomas Wiemann
 */

namespace lvr
{

template<typename VertexT>
SearchTreeNanoflann<VertexT>::SearchTreeNanoflann(
        PointBufferPtr points,
        size_t &n_points,
        const int &kn,
        const int &ki,
        const int &kd,
        const bool &useRansac )
{
    // Build adaptor
    m_pointCloud = new SearchTreeNanoflann<VertexT>::NFPointCloud<float>(points);

    // Build kd-tree
    m_tree = new kd_tree_t(3, *m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(5));
    m_tree->buildIndex();
}


template<typename VertexT>
void SearchTreeNanoflann<VertexT>::kSearch(
           coord < float >& qp,
           int neighbors, vector< ulong > &indices,
           vector< double > &distances )
{

    float query_point[3] = {qp[0], qp[1], qp[2]};
    vector<size_t> ret_index(neighbors);
    vector<float>  dist(neighbors);
    const size_t n = neighbors;
    m_tree->knnSearch(&query_point[0], n, &ret_index[0], &dist[0]);

    for(size_t i = 0; i < ret_index.size(); i++)
    {
        indices.push_back(ret_index[i]);
        distances.push_back(dist[i]);
    }
}
template<typename VertexT>
void SearchTreeNanoflann<VertexT>::kSearch(VertexT qp, int k, vector< VertexT > &nb)
{
    float query_point[3] = {qp[0], qp[1], qp[2]};
    vector<size_t> neighbors(k);
    vector<float> dist(k);
    const size_t n = k;
    m_tree->knnSearch(&query_point[0], n, &neighbors[0], &dist[0]);

    for(size_t i = 0; i < neighbors.size(); i++)
    {
        nb.push_back(VertexT(
                m_pointCloud->m_points[3 * neighbors[i]],
                m_pointCloud->m_points[3 * neighbors[i] + 1],
                m_pointCloud->m_points[3 * neighbors[i] + 2]));
    }
}



template<typename VertexT>
void SearchTreeNanoflann<VertexT>::radiusSearch( float              qp[3], double r, vector< ulong > &indices ) {};

template<typename VertexT>
void SearchTreeNanoflann<VertexT>::radiusSearch( VertexT&              qp, double r, vector< ulong > &indices ) {};

template<typename VertexT>
void SearchTreeNanoflann<VertexT>::radiusSearch( const VertexT&        qp, double r, vector< ulong > &indices ) {};

template<typename VertexT>
void SearchTreeNanoflann<VertexT>::radiusSearch( coord< float >&       qp, double r, vector< ulong > &indices ) {};

template<typename VertexT>
void SearchTreeNanoflann<VertexT>::radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices ) {};






} /* namespace lvr */

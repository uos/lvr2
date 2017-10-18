/*
 * SearchTreeFlann.tcc
 *
 *  Created on: Sep 22, 2015
 *      Author: twiemann
 */

#include "SearchTreeFlann.hpp"

#include <lvr/geometry/VertexTraits.hpp>
#include <lvr/io/Timestamp.hpp>

namespace lvr
{

template<typename VertexT>
SearchTreeFlann< VertexT >::SearchTreeFlann( PointBufferPtr buffer, size_t &n_points, const int &kn, const int &ki, const int &kd )
{
    this->initBuffers(buffer);

    m_flannPoints = flann::Matrix<float> (new float[3 * n_points], n_points, 3);
    for(size_t i = 0; i < n_points; i++)
    {
        m_flannPoints[i][0] = this->m_pointData[3 * i];
        m_flannPoints[i][1] = this->m_pointData[3 * i + 1];
        m_flannPoints[i][2] = this->m_pointData[3 * i + 2];
    }

    m_tree = boost::shared_ptr<flann::Index<flann::L2_Simple<float> > >(new flann::Index<flann::L2_Simple<float> >(m_flannPoints, ::flann::KDTreeSingleIndexParams (10, false)));
    m_tree->buildIndex();

}

template<typename VertexT>
SearchTreeFlann< VertexT >::SearchTreeFlann( std::vector<float> buffer, const int &kn, const int &ki, const int &kd )
{

    m_flannPoints = flann::Matrix<float> (new float[3 * buffer.size()], buffer.size(), 3);
    for(size_t i = 0; i < buffer.size(); i++)
    {
        m_flannPoints[i][0] = buffer[3 * i];
        m_flannPoints[i][1] = buffer[3 * i + 1];
        m_flannPoints[i][2] = buffer[3 * i + 2];
    }

    m_tree = boost::shared_ptr<flann::Index<flann::L2_Simple<float> > >(new flann::Index<flann::L2_Simple<float> >(m_flannPoints, ::flann::KDTreeSingleIndexParams (10, false)));
    m_tree->buildIndex();

}

template<typename VertexT>
SearchTreeFlann< VertexT >::~SearchTreeFlann() {
    delete[] m_flannPoints.ptr();
}

template<typename VertexT>
void  SearchTreeFlann< VertexT >::kSearch( float qx, float qy, float qz, int k, vector< size_t > &indices, vector< float > &distances )
{
    flann::Matrix<float> query_point(new float[3], 1, 3);
    query_point[0][0] = qx;
    query_point[0][1] = qy;
    query_point[0][2] = qz;

    indices.resize(k);
    distances.resize(k);

    flann::Matrix<size_t> ind (&indices[0], 1, k);
    flann::Matrix<float> dist (&distances[0], 1, k);

    flann::SearchParams p;
    m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());
    delete[] query_point.ptr();
}

template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch( coord< float > &qp, int k, vector< int > &indices, vector< float > &distances )
{
    flann::Matrix<float> query_point(new float[3], 1, 3);
    query_point[0][0] = qp.x;
    query_point[0][1] = qp.y;
    query_point[0][2] = qp.z;

    indices.resize(k);
    distances.resize(k);

    flann::Matrix<int> ind (&indices[0], 1, k);
    flann::Matrix<float> dist (&distances[0], 1, k);


    m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
    delete[] query_point.ptr();
}

template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch(VertexT qp, int k, vector< VertexT > &nb)
{
    flann::Matrix<float> query_point(new float[3], 1, 3);
    query_point[0][0] = qp.x;
    query_point[0][1] = qp.y;
    query_point[0][2] = qp.z;

    m_dst.resize(k);
    m_ind.resize(k);

    flann::Matrix<int> ind (&m_ind[0], 1, k);
    flann::Matrix<float> dist (&m_dst[0], 1, k);

    m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());

    for(size_t i = 0; i < k; i++)
    {
        int index = m_ind[i];
        if(index < this->m_numPoints)
        {
            VertexT v(this->m_pointData[3 * index], this->m_pointData[3 * index + 1], this->m_pointData[3 * index + 2]);

            if(this->m_haveColors)
            {
                VertexTraits<VertexT>::setColor(
                        v,
                        this->m_pointColorData[3 * index],
                        this->m_pointColorData[3 * index + 1],
                        this->m_pointColorData[3 * index + 2]);
            }

            nb.push_back(v);
        }
    }
}

/*
   Begin of radiusSearch implementations
 */
template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( float qp[3], float r, vector< int > &indices )
{
    cout << "Flann radius search not yet implemented" << endl;
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( VertexT& qp, float r, vector< int > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const VertexT& qp, float r, vector< int > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( coord< float >& qp, float r, vector< int > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const coord< float >& qp, float r, vector< int > &indices )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->radiusSearch( qp_arr, r, indices );
}

} /* namespace lvr */

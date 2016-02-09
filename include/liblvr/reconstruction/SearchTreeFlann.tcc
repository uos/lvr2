/*
 * SearchTreeFlann.tcc
 *
 *  Created on: Sep 22, 2015
 *      Author: twiemann
 */

#include "SearchTreeFlann.hpp"

namespace lvr
{

template<typename VertexT>
SearchTreeFlann< VertexT >::SearchTreeFlann( PointBufferPtr buffer, size_t &n_points, const int &kn, const int &ki, const int &kd )
{
	m_flannPoints = flann::Matrix<float> (new float[3 * n_points], n_points, 3);
	m_points = buffer->getPointArray(m_numPoints);
	for(size_t i = 0; i < n_points; i++)
	{
		m_flannPoints[i][0] = m_points[3 * i];
		m_flannPoints[i][1] = m_points[3 * i + 1];
		m_flannPoints[i][2] = m_points[3 * i + 2];
	}


	m_tree = boost::shared_ptr<flann::Index<flann::L2_Simple<float> > >(new flann::Index<flann::L2_Simple<float> >(m_flannPoints, ::flann::KDTreeSingleIndexParams (10, false)));
	m_tree->buildIndex();

}


template<typename VertexT>
SearchTreeFlann< VertexT >::~SearchTreeFlann() {

}

template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch( coord< float > &qp, int k, vector< ulong > &indices, vector< float > &distances )
{
	flann::Matrix<float> query_point(new float[3], 1, 3);
	query_point[0][0] = qp.x;
	query_point[0][1] = qp.y;
	query_point[0][2] = qp.z;

	indices.resize(k);
	distances.resize(k);

	flann::Matrix<ulong> ind (&indices[0], 1, k);
	flann::Matrix<float> dist (&distances[0], 1, k);

	m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());

	//for(int i = 0; i < indices.size(); i++) cout << indices[i] << " ";
	//cout << endl;
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

	flann::Matrix<ulong> ind (&m_ind[0], 1, k);
	flann::Matrix<float> dist (&m_dst[0], 1, k);

	m_tree->knnSearch(query_point, ind, dist, k, flann::SearchParams());

	for(size_t i = 0; i < k; i++)
	{
		ulong index = m_ind[i];
		if(index < m_numPoints)
		{
			VertexT v(m_points[3 * index], m_points[3 * index + 1], m_points[3 * index + 2]);
			nb.push_back(v);
		}
	}
}

/*
   Begin of radiusSearch implementations
 */
template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( float qp[3], float r, vector< ulong > &indices )
{
	cout << "Flann radius search not yet implemented" << endl;
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( VertexT& qp, float r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const VertexT& qp, float r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( coord< float >& qp, float r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const coord< float >& qp, float r, vector< ulong > &indices )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->radiusSearch( qp_arr, r, indices );
}

} /* namespace lvr */

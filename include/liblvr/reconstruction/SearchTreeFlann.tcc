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
	size_t n;
	flann::Matrix<float> points(buffer->getPointArray(n).get(), n_points, 3);
	m_tree = boost::shared_ptr<flann::Index<flann::L2_Simple<float> > >(new flann::Index<flann::L2_Simple<float>>(points, flann::AutotunedIndexParams()));
	m_tree->buildIndex();
}


template<typename VertexT>
SearchTreeFlann< VertexT >::~SearchTreeFlann() {
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch( coord< float > &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
	flann::Matrix<float> query_point(new float[3], 1, 3);
	query_point[0][0] = qp.x;
	query_point[0][1] = qp.y;
	query_point[0][2] = qp.z;

	//flann::Matrix<int> ind(new int[3 * neighbors], neighbors, 3);
	//flann::Matrix<float> dist(new float[3 * neighbors], neighbors, 3);

	m_tree->knnSearch(query_point, indices, distances, neighbors, flann::SearchParams());
}

template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch(VertexT qp, int k, vector< VertexT > &neighbors)
{
	flann::Matrix<float> query_point(new float[3], 1, 3);
	query_point[0][0] = qp[0];
	query_point[0][1] = qp[1];
	query_point[0][2] = qp[2];

	flann::Matrix<int> ind(new int[k], 1, k);
	flann::Matrix<float> dist(new float[k], 1, k);

	m_tree->knnSearch(query_point, ind, dist, neighbors, flann::SearchParams());

	for(size_t i = 0; i < k; i++)
	{
		size_t index = ind[0][1];
		neighbors.push_back(VertexT(m_points[3 * index], m_points[3 * index + 1], m_points[3 * index + 2]));
	}
}

/*
   Begin of radiusSearch implementations
 */
template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( float qp[3], double r, vector< ulong > &indices )
{
    // TODO: Implement me!
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->radiusSearch( qp_arr, r, indices );
}

} /* namespace lvr */

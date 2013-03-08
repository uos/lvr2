/*
 * KdTree.hpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

#ifndef KDTREE_HPP_
#define KDTREE_HPP_

#include "geometry/BoundingBox.hpp"
#include "io/ModelFactory.hpp"
#include "io/PointBuffer.hpp"
#include "geometry/Vertex.hpp"
#include "io/Model.hpp"

#include "io/PLYIO.hpp"
#include "io/AsciiIO.hpp"
#include "io/UosIO.hpp"

// vorerst
#include <cstring>
#include <iostream>
#include <list>
#include "KdNode.hpp"
#include "boost/shared_array.hpp"

namespace lssr{

/**
 * @brief A class for execution of a distribution using a very simple Kd-tree.
 * 	The classification criterion is to halve the longest axis.
 */
template<typename VertexT>
class KdTree {
public:

	/**
	 * @brief Test Construktor
	 */
	KdTree();


	virtual ~KdTree();

	/**
	 * @brief Constructor.
	 *
	 * @param pointcloud The point cloud, which is to be divided
	 */
	KdTree(PointBufferPtr pointcloud, long int max);

	/**
	 * @brief starts the recursion and saves the packages in files (scan*.3d)
	 *
	 * @param first The first Node with the whole pointcloud in it
	 */
	void Rekursion(KdNode<VertexT> * first);


	/**
	 * @brief This function devide the pointcloud in two packages.
	 *
	 * 	The classification criterion is to halve the longest axis.
	 *	When a partial pointcloud has less than the maximum number of points is thus, it will be stored in a 3D file.
	 */
	void splitPointcloud(KdNode<VertexT> * child);

	/**
	 * @brief returns the list of kdNodes
	 */
	list<KdNode<VertexT>*> GetList();


	// The pointcloud
	PointBufferPtr         m_loader;

	/// The currently stored points
	coord3fArr   			m_points;

    /// The bounding box of the point cloud
    BoundingBox<VertexT>   m_boundingBox;

    // A list for all Nodes with less than MAX_POINTS
    std::list<KdNode<VertexT>*> nodelist;

    // number of max Points in one packete
    long int max_points;



	// Number of points in the point cloud
	size_t m_numpoint;

	// Number of points per packet
	double m_bucketsize;

	// A shared-Pointer for the model, with the pointcloud in it  */
	ModelPtr m_model;
};
}

#include "KdTree.tcc"
#endif /* KDTREE_HPP_ */

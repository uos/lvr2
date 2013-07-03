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
#include <cctype>
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

	virtual ~KdTree();

	/**
	 * @brief Constructor.
	 *
	 * @param pointcloud The point cloud, which is to be divided
	 * @param max        Max number of Points
	 * @param min        Min number of Points
	 * @param median     is True, if the median is used for segmenting the pointcloud
	 */
	KdTree(PointBufferPtr pointcloud, long int max, long int min, bool median);

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
	 *
	 *@param child Instance KdNode 
	 *@param again count if this is the second call with the same KdNode, if it is take another axis
	 */
	void splitPointcloud(KdNode<VertexT> * child, int again);

	/**
	 * @brief returns the list of kdNodes
	 */
	list<KdNode<VertexT>*> GetList();
	

	/**
	 *@brief function to sort list
	 */
	bool static compare_nocase (KdNode<VertexT> * first, KdNode<VertexT> * second);	

	/**
	 * @brief return the BoundingBox
	 */
	BoundingBox<VertexT> GetBoundingBox();


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
  
    // number of min Points in one packete
    long int min_points;


	// Number of points in the point cloud
	size_t m_numpoint;

	//use median
	bool m_median;

	// Number of points per packet
	double m_bucketsize;

	// A shared-Pointer for the model, with the pointcloud in it  */
	ModelPtr m_model;
};
}

#include "KdTree.tcc"
#endif /* KDTREE_HPP_ */

/*
 * MPITree.hpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

#ifndef LARGEKDTREE_HPP_
#define LARGEKDTREE_HPP_



namespace lvr{

/**
 * @brief A class for execution of a distribution using a very simple Kd-tree.
 * 	The classification criterion is to halve the longest axis.
 */
class LargeKDTree {
public:

	virtual ~LargeKDTree();

	/**
	 * @brief Constructor.
	 *
	 * @param pointcloud The point cloud, which is to be divided
	 * @param max        Max number of Points
	 * @param min        Min number of Points
	 * @param median     is True, if the median is used for segmenting the pointcloud
	 */
   LargeKDTree(PointBufferPtr pointcloud, long int max, long int min, bool median);

	/**
	 * @brief starts the recursion and saves the packages in files (scan*.3d)
	 *
	 * @param first The first Node with the whole pointcloud in it
	 */
	void Rekursion(MPINode<VertexT> * first);


	/**
	 * @brief This function devide the pointcloud in two packages.
	 *
	 * 	The classification criterion is to halve the longest axis.
	 *	When a partial pointcloud has less than the maximum number of points is thus, it will be stored in a 3D file.
	 *
	 *@param child Instance MPINode
	 *@param again count if this is the second call with the same MPINode, if it is take another axis
	 */
	void splitPointcloud(MPINode<VertexT> * child, int again);

	/**
	 * @brief returns the list of MPINodes
	 */
	list<MPINode<VertexT>*> GetList();


	/**
	 *@brief function to sort list
	 */
	bool static compare_nocase (MPINode<VertexT> * first, MPINode<VertexT> * second);

	/**
	 * @brief return the BoundingBox
	 */
	BoundingBox<VertexT> GetBoundingBox();



};
}

#endif /* LARGEKDTREE_HPP_ */

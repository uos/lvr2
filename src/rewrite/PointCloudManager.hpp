/*
 * PointCloudManager.h
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUDMANAGER_H_
#define POINTCLOUDMANAGER_H_

#include <vector>
using std::vector;

#include "Vertex.hpp"
#include "Normal.hpp"
#include "BoundingBox.hpp"

namespace lssr
{

/**
 * @brief	Abstract interface class for objects that are
 * 			able to handle point cloud data with normals. It
 * 			defines queries for nearest neighbor search.
 */
template<typename T>
class PointCloudManager
{
public:
	/**
	 * @brief Returns the k closest neighbor vertices to a given query point
	 *
	 * @param v			A query vertex
	 * @param k			The (max) number of returned closest points to v
	 * @param nb		A vector containing the determined closest points
	 */
	virtual void getkClosestVertices(const Vertex<T> &v,
		const size_t &k, vector<Vertex<T> > &nb) = 0;

	/**
	 * @brief Returns the k closest neighbor normals to a given query point
	 *
	 * @param n			A query vertex
	 * @param k			The (max) number of returned closest points to v
	 * @param nb		A vector containing the determined closest normals
	 */
	virtual void getkClosestNormals(const Vertex<T> &n,
		const size_t &k, vector<Normal<T> > &nb) = 0;

	/**
	 * @brief Returns the bounding box of the loaded point set
	 */
	virtual BoundingBox<T>& getBoundingBox();

protected:

    /// The currently stored points
    T**                         m_points;

    /// The point normals
    T**                         m_normals;

    /// The bounding box of the point set
    BoundingBox<T>              m_boundingBox;
};

} // namespace lssr

#include "PointCloudManager.tcc"

#endif /* POINTCLOUDMANAGER_H_ */

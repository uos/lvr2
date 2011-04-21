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

#include "../geometry/Vertex.hpp"
#include "../geometry/Normal.hpp"
#include "../geometry/BoundingBox.hpp"
#include "../io/AsciiIO.hpp"

namespace lssr
{

/**
 * @brief	Abstract interface class for objects that are
 * 			able to handle point cloud data with normals. It
 * 			defines queries for nearest neighbor search.
 */
template<typename VertexT, typename NormalT>
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
	virtual void getkClosestVertices(const VertexT &v,
		const size_t &k, vector<VertexT> &nb) = 0;

	/**
	 * @brief Returns the k closest neighbor normals to a given query point
	 *
	 * @param n			A query vertex
	 * @param k			The (max) number of returned closest points to v
	 * @param nb		A vector containing the determined closest normals
	 */
	virtual void getkClosestNormals(const VertexT &n,
		const size_t &k, vector<NormalT> &nb) = 0;

	/**
	 * @brief Returns the bounding box of the loaded point set
	 */
	virtual BoundingBox<VertexT>& getBoundingBox();

	/**
	 * @brief Returns the points at index \ref{index} in the point array
	 *
	 * @param index
	 * @return
	 */
	virtual VertexT* getPoint(size_t index);

	/**
	 * @brief Returns the number of managed points
	 */
	virtual size_t getNumPoints();

	/**
	 * @brief Returns the point at the given \ref{index}
	 */
	virtual const VertexT* operator[](const size_t &index) const;

	virtual float distance(VertexT v) = 0;

protected:

	/**
	 * @brief Tries to read point and normal information from
	 *        the given file
	 *
	 * @param filename      A file containing point cloud data.
	 */
	virtual void readFromFile(string filename);

    /// The currently stored points
    float**                   	m_points;

    /// The point normals
    float**                  	m_normals;

    /// The bounding box of the point set
    BoundingBox<VertexT>        m_boundingBox;

    size_t                      m_numPoints;
};

} // namespace lssr

#include "PointCloudManager.tcc"

#endif /* POINTCLOUDMANAGER_H_ */

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


 /*
 * PointCloudManager.h
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUDMANAGER_H_
#define POINTCLOUDMANAGER_H_

#include <vector>


#include "../geometry/Vertex.hpp"
#include "../geometry/Normal.hpp"
#include "../geometry/BoundingBox.hpp"

using std::vector;

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
	virtual VertexT getPoint(size_t index);

	/**
	 * @brief Returns the number of managed points
	 */
	virtual size_t getNumPoints();

	/**
	 * @brief Returns the point at the given \ref{index}
	 */
	virtual const VertexT operator[](const size_t &index) const;

	virtual float distance(VertexT v) = 0;

	void setKD(int kd) {m_kd = kd;}

	void setKI(int ki) {m_ki = ki;}

	void setKN(int kn) {m_kn = kn;}

	virtual void calcNormals() = 0;

protected:

    /// The currently stored points
    float**                   	m_points;

    /// The point normals
    float**                  	m_normals;

    /// Color information for points
    uint8_t **                  m_colors;

    /// The bounding box of the point set
    BoundingBox<VertexT>        m_boundingBox;

    size_t                      m_numPoints;

    /// The number of neighbors used for initial normal estimation
    int                         m_kn;

    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    /// The number of tangent planes used for distance determination
    int                         m_kd;
};

} // namespace lssr

#include "PointCloudManager.tcc"

#endif /* POINTCLOUDMANAGER_H_ */

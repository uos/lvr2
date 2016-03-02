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
 * BoundingBox.hpp
 *
 *  @date 22.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef BOUNDINGBOX_H_
#define BOUNDINGBOX_H_

#include <algorithm>
#include <cmath>
#include <limits>
using std::numeric_limits;

#include "Vertex.hpp"

namespace lvr
{

/**
 * @brief A dynamic bounding box class.
 */
template<typename VertexT>
class BoundingBox {
public:

    /**
     * @brief Default constructor
     */
	BoundingBox();

	/**
	 * @brief Constructs a bounding box with from the given vertices
	 *
	 * @param v1        Lower left corner of the BoundingBox
	 * @param v2        Upper right corner of the BoundingBox
	 * @return
	 */
	BoundingBox(VertexT v1, VertexT v2);

	/**
	 * @brief Constructs a bounding box from the given coordinates
	 *
	 * @param x_min     x value of the lower left corner
	 * @param y_min     y value of the lower left corner
	 * @param z_min     z value of the lower left corner
	 * @param x_max     x value of the upper right corner
	 * @param y_max     y value of the upper right corner
	 * @param z_max     z value of the upper right corner
	 * @return
	 */
	BoundingBox(float x_min, float y_min, float z_min,
			    float x_max, float y_max, float z_max);

	virtual ~BoundingBox() {};

	/**
	 * @brief Expands the bounding box if the given point \ref{v} is
	 *        outside the current volume
	 *
	 * @param v         A 3d point
	 */
	inline void         expand(VertexT v);

	/**
	 * @brief Expands the bounding box if the given point is outside
     *        the current volume
     *
	 * @param x         The x coordinate of the check point
	 * @param y         The y coordinate of the check point
	 * @param z         The z coordinate of the check point
	 */
	inline void         expand(float x, float y, float z);

	/**
	 * @brief  Calculates the surrounding bounding box of the current
	 *         volume and the other given bounding box
	 *
	 * @param bb        Another bounding box
	 */
	inline void         expand(BoundingBox<VertexT>& bb);

	/**
	 * @brief Returns the radius of the current volume, i.e. the distance
	 *        between the centroid and the most distance corner from this
	 *        point.
	 */
	float 		        getRadius();

	/**
	 * @brief Returns true if the bounding box has been expanded before or
	 *        was initialized with a preset size.
	 */
	bool                isValid();

	/**
	 * @brief Returns the center point of the bounding box.
	 */
	VertexT           	getCentroid(){return m_centroid;};

	/**
	 * @brief Returns the longest side of the bounding box
	 */
	float               getLongestSide();

	/**
	 * @brief Returns the x-size of the bounding box
	 */
	float               getXSize();

    /**
     * @brief Returns the y-size of the bounding box
     */
    float               getYSize();

    /**
     * @brief Returns the z-size of the bounding box
     */
    float               getZSize();


    /**
     * @brief Returns the upper right coordinates
     */
    VertexT           	getMax() const;

    /**
     * @brief Returns the lower left coordinates
     */
    VertexT             getMin() const;

private:

	/// The lower right point of the bounding box
	VertexT           	m_min;

	/// The upper right point of the bounding box
	VertexT           	m_max;

	/// The center point of the bounding box
	VertexT           	m_centroid;

	/// The 'width' of the bounding box
	float               m_xSize;

	/// The 'height' of the bounding box
	float               m_ySize;

	/// The 'depth' of the bounding box
	float               m_zSize;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, BoundingBox<T>&bb)
{
    os << "Bounding Box: " << endl;
    os << "Min \t\t: " << bb.getMin();
    os << "Max \t\t: " << bb.getMax();
    os << "Dimensions \t: " << bb.getXSize() << " " << bb.getYSize() << " " << bb.getZSize();
    return os;
}

} // namespace lvr

#include "BoundingBox.tcc"

#endif /* BOUNDINGBOX_H_ */

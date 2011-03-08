
/*
 * BoundingBox.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef BOUNDINGBOX_H_
#define BOUNDINGBOX_H_

#include <algorithm>
#include <cmath>
#include <limits>
using std::numeric_limits;

#include "Vertex.hpp"

namespace lssr
{

/***
 * @brief A dynamic bounding box class for point sets.
 */
template<typename T>
class BoundingBox {
public:

    /**
     * @brief Defaulkt constructor
     */
	BoundingBox();

	/**
	 * @brief Constructs a bounding box with from the given vertices
	 *
	 * @param v1        Lower left corner of the BoundingBox
	 * @param v2        Upper right corner of the BoundingBox
	 * @return
	 */
	BoundingBox(Vertex<T> v1, Vertex<T> v2);

	/**
	 * @brief Constrcuts a bounding box from the given coordinates
	 *
	 * @param x_min     x value of the lower left corner
	 * @param y_min     y value of the lower left corner
	 * @param z_min     z value of the lower left corner
	 * @param x_max     x value of the upper right corner
	 * @param y_max     y value of the upper right corner
	 * @param z_max     z value of the upper right corner
	 * @return
	 */
	BoundingBox(T x_min, T y_min, T z_min,
			    T x_max, T y_max, T z_max);

	virtual ~BoundingBox() {};

	/**
	 * @brief Expands the bounding box if the given point \ref{v} is
	 *        outside the current volume
	 *
	 * @param v         A 3d point
	 */
	inline void         expand(Vertex<T> v);

	/**
	 * @brief Expands the bounding box if the given point is outside
     *        the current volume
     *
	 * @param x         The x coordinate of the check point
	 * @param y         The y coordinate of the check point
	 * @param z         The z coordinate of the check point
	 */
	inline void         expand(T x, T y, T z);

	/**
	 * @brief  Calculates the surrounding bounding box of the current
	 *         volume and the other given bounding box
	 *
	 * @param bb        Another bounding box
	 */
	inline void         expand(BoundingBox<T>& bb);

	/**
	 * @brief Returns the radius of the current volume, i.e. the distance
	 *        between the centroid and the most distance corner from this
	 *        point.
	 */
	T 		            getRadius();

	/**
	 * @brief Returns true if the bounding box has been expanded before or
	 *        was initialized with a preset size.
	 */
	bool                isValid();

	/**
	 * @brief Returns the center point of the bounding box.
	 */
	Vertex<T>           getCentroid(){return m_centroid;};

	/**
	 * @brief Returns the longest side of the bounding box
	 */
	T                   getLongestSide();

	/**
	 * @brief Returns the x-size of the bounding box
	 */
	T                   getXSize();

    /**
     * @brief Returns the y-size of the bounding box
     */
    T                   getYSize();

    /**
     * @brief Returns the z-size of the bounding box
     */
    T                   getZSize();


    /**
     * @brief Returns the upper right coordinates
     */
    Vertex<T>           getMax() const;

    /**
     * @brief Returns the lower left coordinates
     */
    Vertex<T>           getMin() const;

private:

	/// The lower right point of the bounding box
	Vertex<T>           m_min;

	/// The upper right point of the bounding box
	Vertex<T>           m_max;

	/// The center point of the bounding box
	Vertex<T>           m_centroid;

	/// The 'width' of the bounding box
	T                   m_xSize;

	/// The 'height' of the bounding box
	T                   m_ySize;

	/// The 'depth' of the bounding box
	T                   m_zSize;
};

} // namespace lssr

#include "BoundingBox.tcc"

#endif /* BOUNDINGBOX_H_ */

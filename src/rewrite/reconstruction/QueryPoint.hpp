/*
 * QueryPoint.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef QUERYPOINT_H_
#define QUERYPOINT_H_

namespace lssr
{

/**
 * @brief A query point for marching cubes reconstructions.
 *        It represents a point in space together with a
 *        'distance' value that is used by the marching
 *        cubes algorithm
 */
template<typename T>
class QueryPoint {
public:

    /**
     * @brief Default constructor.
     */
	QueryPoint();

	/**
	 * @brief Constructor.
	 *
	 * @param v          The position of the query point. The distance
	 *                   value is set to 0
	 */
	QueryPoint(Vertex<T> v);

	/**
	 * @brief Constructor.
	 *
	 * @param v         The position of the query point.
	 * @param f         The distance value for the query point.
	 */
	QueryPoint(Vertex<T> v, T f);

	/**
	 * @brief Copy constructor.
	 * @param o
	 * @return
	 */
	QueryPoint(const QueryPoint &o);

	/**
	 * @brief Destructor.
	 */
	virtual ~QueryPoint() {};

	/// The position of the query point
	Vertex<T>       m_position;

	/// The associated distance value
	T               m_distance;
};

} // namespace lssr

#include "QueryPoint.tcc"

#endif /* QUERYPOINT_H_ */

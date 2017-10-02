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
 * QueryPoint.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_QUERYPOINT_H_
#define _LVR2_RECONSTRUCTION_QUERYPOINT_H_

#include <lvr2/geometry/Point.hpp>

namespace lvr2
{

/**
 * @brief A query point for marching cubes reconstructions.
 *        It represents a point in space together with a
 *        'distance' value that is used by the marching
 *        cubes algorithm
 */
template<typename BaseVecT>
class QueryPoint
{
public:

    /**
     * @brief Default constructor.
     */
    QueryPoint();

    /**
     * @brief Constructor.
     *
     * @param p          The position of the query point. The distance
     *                   value is set to 0
     */
    QueryPoint(Point<BaseVecT> p);

    /**
     * @brief Constructor.
     *
     * @param p         The position of the query point.
     * @param f         The distance value for the query point.
     */
    QueryPoint(Point<BaseVecT> p, float f);

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
    Point<BaseVecT> m_position;

    /// The associated distance value
    float           m_distance;

    /// Indicates if the query point is valid
    bool            m_invalid;
};

} // namespace lvr2

#include "QueryPoint.tcc"

#endif /* _LVR2_RECONSTRUCTION_QUERYPOINT_H_ */
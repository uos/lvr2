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
 * Plane.hpp
 *
 *  @date 14.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_GEOMETRY_PLANE_H_
#define LVR2_GEOMETRY_PLANE_H_

#include "Normal.hpp"
#include "Point.hpp"

namespace lvr2
{

/**
 * @brief A plane.
 *
 * A plane represented by a normal and a position vector.
 */
template <typename BaseVecT>
struct Plane
{
    Plane() : normal(0, 0, 1) {}

    Normal<BaseVecT> normal;
    Point<BaseVecT> pos;

    /// Projects the given point onto the plane and returns the projection point.
    Point<BaseVecT> project(const Point<BaseVecT>& other) const;

    /**
     * @brief Calculates the distance between the plane and the given point.
     * @return This can be < 0, == 0 or > 0 the cases mean:
     *         < 0: The point lies between the plane and the origin
     *         == 0: The point lies in the plane
     *         > 0: The point lies behind the plane, oberserved from the origin
     */
    float distance(const Point<BaseVecT>& other) const;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const Plane<BaseVecT>& p)
{
    os << "Plane[" << p.normal << ", " << p.pos << "]";
    return os;
}

} // namespace lvr2

#include <lvr2/geometry/Plane.tcc>

#endif /* LVR2_GEOMETRY_PLANE_H_ */

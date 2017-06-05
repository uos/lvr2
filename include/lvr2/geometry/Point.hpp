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
 * Point.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_POINT_H_
#define LVR2_GEOMETRY_POINT_H_


namespace lvr2
{

template <typename> struct Vector;

template <typename BaseVecT>
struct Point : public BaseVecT
{
    Point() {}
    Point(BaseVecT base) : BaseVecT(base) {}


    // It doesn't make sense to talk about the length of a position vector.
    // It's the same as asking: "What is the length of New York?".
    typename BaseVecT::CoordType length() const = delete;
    typename BaseVecT::CoordType length2() const = delete;

    // Similarly, the cross product only makes sense for direction vectors.
    BaseVecT cross(const BaseVecT &other) const = delete;

    // The standard operators are deleted and replaced by strongly typed ones.
    BaseVecT operator+(const BaseVecT &other) const = delete;
    BaseVecT operator-(const BaseVecT &other) const = delete;
    BaseVecT& operator-=(const BaseVecT &other) = delete;
    BaseVecT& operator+=(const BaseVecT &other) = delete;

    // Adding/subtracting a point and a vector makes sense and results in a
    // new point.
    Point<BaseVecT> operator+(const Vector<BaseVecT> &other) const;
    Point<BaseVecT> operator-(const Vector<BaseVecT> &other) const;
    Point<BaseVecT>& operator+=(const Vector<BaseVecT> &other);
    Point<BaseVecT>& operator-=(const Vector<BaseVecT> &other);
};

} // namespace lvr

#include "Point.tcc"

#endif /* LVR2_GEOMETRY_POINT_H_ */

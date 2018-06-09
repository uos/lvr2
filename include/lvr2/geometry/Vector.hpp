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
 * Vector.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_VECTOR_H_
#define LVR2_GEOMETRY_VECTOR_H_

#include <ostream>


namespace lvr2
{

// Forward declarations
template <typename> struct Normal;
template <typename> struct Point;

/**
 * @brief A strongly-typed vector, representing a direction vector.
 *
 * This type represents a direction vector as opposed to a position vector.
 * This is a subtle but important difference. There are operations that don't
 * make sense for certain combinations of position/direction vectors. In order
 * to avoid logic bugs, this type and `Point` delete certain operations from
 * the backing type.
 *
 * For more information and intuitive examples why this type safety makes
 * sense, see [1].
 *
 *
 * [1]: https://math.stackexchange.com/a/645827/340615
 */
template <typename BaseVecT>
struct Vector : public BaseVecT
{
    Vector() {}
    Vector(BaseVecT base) : BaseVecT(base) {}
    using BaseVecT::BaseVecT;

    /**
     * @brief Returns a normalized version of this vector.
     *
     * Note that `this` must not be the null vector, or else the behavior is
     * undefined.
     */
    Normal<BaseVecT> normalized() const;

    /**
     * @brief Normalizes the vector in place.
     *
     * Consider using `normalized()` and the `Normal` type to increase type
     * safety when dealing with normalized vectors.
     *
     * Note that `this` must not be the null vector, or else the behavior is
     * undefined.
     */
    void normalize();


    /**
     * @brief Returns the average of all vectors in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Vector<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static Vector<BaseVecT> average(const CollectionT& vecs);

    // It doesn't make sense to talk about the distance between two direction
    // vectors. It's the same as asking: "What is the distance between
    // '3 meters north' and '10 cm east'.
    typename BaseVecT::CoordType distance(const BaseVecT &other) const = delete;
    typename BaseVecT::CoordType distance2(const BaseVecT &other) const = delete;

    // More type safe overwrite
    Vector<BaseVecT> cross(const Vector<BaseVecT> &other) const;

    // The standard operators are deleted and replaced by strongly typed ones.
    BaseVecT operator+(const BaseVecT &other) const = delete;
    BaseVecT operator-(const BaseVecT &other) const = delete;
    BaseVecT& operator-=(const BaseVecT &other) = delete;
    BaseVecT& operator+=(const BaseVecT &other) = delete;

    // Addition/subtraction between two vectors
    Vector<BaseVecT> operator+(const Vector<BaseVecT> &other) const;
    Vector<BaseVecT> operator-(const Vector<BaseVecT> &other) const;
    Vector<BaseVecT>& operator+=(const Vector<BaseVecT> &other);
    Vector<BaseVecT>& operator-=(const Vector<BaseVecT> &other);

    /// Scalar multiplication.
    Vector<BaseVecT> operator*(const typename BaseVecT::CoordType &scale) const;
    /// Scalar division.
    Vector<BaseVecT> operator/(const typename BaseVecT::CoordType &scale) const;

    // Addition/subtraction between point and vector
    Point<BaseVecT> operator+(const Point<BaseVecT> &other) const;
    Point<BaseVecT> operator-(const Point<BaseVecT> &other) const;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const Vector<BaseVecT>& v)
{
    os << "Vector[" << v.x << ", " << v.y << ", " << v.z << "]";
    return os;
}

} // namespace lvr2

#include <lvr2/geometry/Vector.tcc>

#endif /* LVR2_GEOMETRY_VECTOR_H_ */

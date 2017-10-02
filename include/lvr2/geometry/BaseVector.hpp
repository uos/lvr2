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
 * BaseVector.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_BASEVECTOR_H_
#define LVR2_GEOMETRY_BASEVECTOR_H_


namespace lvr2
{

/**
 * @brief A generic, weakly-typed vector.
 *
 * This vector is weakly-typed as it allows all common operations that can be
 * executed numerically, including operations that won't make sense
 * semantically. You're advised to use `Point` and `Vector` instead of
 * this type directly.
 *
 * This type is used as "backing type" meaning that it is the actual
 * implementation of the vector type. `Vector` and `Point` are just wrappers
 * which delete and/or modify a few methods to make use of strong typing.
 *
 * Instead of this `BaseVector` implementations from other libraries could be
 * used. However, they have to provide all the methods and fields that
 * `BaseVector` defines.
 */
template <typename CoordT>
struct BaseVector
{
public:
    using CoordType = CoordT;

    CoordT x;
    CoordT y;
    CoordT z;

    /// Default constructs a null-vector.
    BaseVector() : x(0), y(0), z(0) {}

    /// Builds a BaseVector with the given coordinates.
    BaseVector(const CoordT &x, const CoordT &y, const CoordT &z)
        : x(x), y(y), z(z)
    {
    }

    // ========================================================================
    // === Named operations
    // ========================================================================

    /**
     * @brief  Returns the length of this vector.
     */
    CoordT length() const;

    /**
     * @brief  Returns the squared length of this vector.
     *
     * The squared length is easier to calculate and sufficient for certain
     * uses cases. This method only exists for performance reasons.
     */
    CoordT length2() const;

    /**
     * @brief  Calculates the distance to another vector.
     */
    CoordT distance(const BaseVector &other) const;

    /**
     * @brief  Calculates the squared distance to another vector.
     *
     * The squared distance is easier to calculate and sufficient for certain
     * uses cases. This method only exists for performance reasons.
     */
    CoordT distance2(const BaseVector &other) const;

    /**
     * @brief    Calculates the cross product between this and
     *           the given vector. Returns a new BaseVector instance.
     */
    BaseVector<CoordT> cross(const BaseVector &other) const;

    /**
     * @brief    Calculates the dot product between this and
     *           the given vector.
     */
    CoordT dot(const BaseVector &other) const;


    // ========================================================================
    // === Operator overloads
    // ========================================================================

    /// Scalar multiplication.
    BaseVector<CoordT> operator*(const CoordT &scale) const;
    /// Scalar division.
    BaseVector<CoordT> operator/(const CoordT &scale) const;

    /// Scalar multiplication.
    BaseVector<CoordT>& operator*=(const CoordT &scale);
    /// Scalar division.
    BaseVector<CoordT>& operator/=(const CoordT &scale);

    /// Element-wise addition.
    BaseVector<CoordT> operator+(const BaseVector &other) const;
    /// Element-wise subtraction.
    BaseVector<CoordT> operator-(const BaseVector &other) const;

    /// Element-wise addition.
    BaseVector<CoordT>& operator+=(const BaseVector<CoordT> &other);
    /// Element-wise subtraction.
    BaseVector<CoordT>& operator-=(const BaseVector<CoordT> &other);

    bool operator==(const BaseVector &other) const;
    bool operator!=(const BaseVector &other) const;
};

} // namespace lvr

#include <lvr2/geometry/BaseVector.tcc>

#endif /* LVR2_GEOMETRY_BASEVECTOR_H_ */
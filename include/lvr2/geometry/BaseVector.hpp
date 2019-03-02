/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * BaseVector.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_BASEVECTOR_H_
#define LVR2_GEOMETRY_BASEVECTOR_H_

#include <iostream>

namespace lvr2
{

template <typename> struct Normal;

/**
 * @brief A generic, weakly-typed vector.
 *
 * This vector is weakly-typed as it allows all common operations that can be
 * executed numerically, including operations that won't make sense
 * semantically. You're advised to use `Point` and `Vector` instead of
 * this type directly.
 *
 * This type is used as "backing type", meaning that it is the actual
 * implementation of the vector type. `Vector` and `Point` are just wrappers
 * which delete and/or modify a few methods to make use of strong typing.
 *
 * Instead of this, `BaseVector` implementations from other libraries could be
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

    BaseVector(const BaseVector& o) : x(o.x), y(o.y), z(o.z)
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

   
    void normalize()
    {
        // Check for invalid vector. This check can be disabled in release mode,
        // which will probably lead to +inf and -inf values. In the documentation
        // for this function we require the vector to not be the null-vector.
        // assert(!(this->x == 0 && this->y == 0 && this->z == 0));
        if(!(this->x == 0 && this->y == 0 && this->z == 0))
        {
            auto len = this->length();
            this->x /= len;
            this->y /= len;
            this->z /= len;
        }
    }


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

    CoordType distanceFrom(const BaseVector<CoordT> &other) const;
    CoordType squaredDistanceFrom(const BaseVector<CoordType> &other) const;

      /**
     * @brief Returns the centroid of all points in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Point<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static BaseVector<CoordT> centroid(const CollectionT& points);


    /**
     * @brief Returns the average of all vectors in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Vector<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static BaseVector<CoordT> average(const CollectionT& vecs);

        /**
     * @brief Returns a normalized version of this vector.
     *
     * Note that `this` must not be the null vector, or else the behavior is
     * undefined.
     */
    Normal<CoordT> normalized() const;

    bool operator==(const BaseVector &other) const;
    bool operator!=(const BaseVector &other) const;

    CoordT operator*(const BaseVector<CoordType> &other) const;

    /**
     * @brief    Indexed coordinate access (reading)
     */
    CoordT operator[](const unsigned& index) const;

    /**
     * @brief   Indexed coordinate access (writing)
     */
    CoordT& operator[](const unsigned& index);
};

template<typename T>
std::ostream& operator<<( std::ostream& os, const BaseVector<T>& v)
{
    os << "Vec: [" << v.x << " " << v.y << " " << v.z << "]" << std::endl;
    return os;
}

} // namespace lvr

#include <lvr2/geometry/BaseVector.tcc>

#endif /* LVR2_GEOMETRY_BASEVECTOR_H_ */

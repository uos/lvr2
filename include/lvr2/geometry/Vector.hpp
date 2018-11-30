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
     * @brief Returns the centroid of all points in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Point<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static Vector<BaseVecT> centroid(const CollectionT& points);


    /**
     * @brief Returns the average of all vectors in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Vector<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static Vector<BaseVecT> average(const CollectionT& vecs);


    // Calculate the distance between this point and the given point.
    typename BaseVecT::CoordType distanceFrom(const Vector<BaseVecT> &other) const;
    typename BaseVecT::CoordType squaredDistanceFrom(const Vector<BaseVecT> &other) const;

    // More type safe overwrite
    Vector<BaseVecT> cross(const Vector<BaseVecT> &other) const;

    // Addition/subtraction between two vectors
    Vector<BaseVecT> operator+(const Vector<BaseVecT> &other) const;
    Vector<BaseVecT> operator-(const Vector<BaseVecT> &other) const;
    Vector<BaseVecT>& operator+=(const Vector<BaseVecT> &other);
    Vector<BaseVecT>& operator-=(const Vector<BaseVecT> &other);

    /// Scalar multiplication.
    Vector<BaseVecT> operator*(const typename BaseVecT::CoordType &scale) const;

    /// Scalar division.
    Vector<BaseVecT> operator/(const typename BaseVecT::CoordType &scale) const;

    typename BaseVecT::CoordType operator*(const Vector<BaseVecT> &other) const
    {
        return this->dot(other);
    }
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

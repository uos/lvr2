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

// Eigen sometimes produces errors when compiled with CUDA. Disables
// all Eigen related function for CUDA code (which is currently fine).
#ifndef __NVCC__
#include <Eigen/Dense>
#endif

#include <iostream>

namespace lvr2
{

template <typename> struct Normal;

/**
 * @brief A generic, weakly-typed vector.
 *
 * This vector is weakly-typed as it allows all common operations that can be
 * executed numerically. Instead of this, `BaseVector` implementations from other libraries could be
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
     * @brief    Calculates the rotated vector around an normal vector n with the rotation angle alpha
     *
     */
    BaseVector<CoordT> rotated(const BaseVector &n, const double &alpha) const;

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

// Eigen sometimes produces errors when compiled with CUDA. Disables
// all Eigen related function for CUDA code (which is currently fine).
#ifndef __NVCC__
    // Friend declaration for Eigen multiplication
    template<typename T, typename S>
    friend BaseVector<T> operator*(const Eigen::Matrix<S, 4, 4>& mat, const BaseVector<T>& normal);

#endif // ifndef __NVCC__

};

template<typename T>
std::ostream& operator<<( std::ostream& os, const BaseVector<T>& v)
{
    os << "Vec: [" << v.x << " " << v.y << " " << v.z << "]" << std::endl;
    return os;
}

// Eigen sometimes produces errors when compiled with CUDA. Disables
// all Eigen related function for CUDA code (which is currently fine).
#ifndef __NVCC__

/**
 * @brief   Multiplication operator to support transformation with Eigen
 *          matrices. Rotates the normal, ignores translation. Implementation
 *          for RowMajor matrices.
 * 
 * @tparam CoordType            Coordinate type of the normals
 * @tparam Scalar               Scalar type of the Eigen matrix
 * @param mat                   Eigen matrix 
 * @param normal                Input normal
 * @return Normal<CoordType>    Transformed normal
 */
template<typename CoordType, typename Scalar = CoordType>
inline BaseVector<CoordType> operator*(const Eigen::Matrix<Scalar, 4, 4>& mat, const BaseVector<CoordType>& normal)
{
    // TODO: CHECK IF THIS IS CORRECT
    CoordType x = mat(0, 0) * normal.x + mat(1, 0) * normal.y + mat(2, 0) * normal.z;
    CoordType y = mat(0, 1) * normal.x + mat(1, 1) * normal.y + mat(2, 1) * normal.z;
    CoordType z = mat(0, 2) * normal.x + mat(1, 2) * normal.y + mat(2, 2) * normal.z;

    x += mat(0, 3);
    y += mat(1, 3);
    z += mat(2, 3);

    return BaseVector<CoordType>(x,y,z);
}

#endif // ifndef __NVCC__

} // namespace lvr

#include "lvr2/geometry/BaseVector.tcc"

#endif /* LVR2_GEOMETRY_BASEVECTOR_H_ */

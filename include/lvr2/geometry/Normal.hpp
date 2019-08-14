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
 * Normal.hpp
 *
 *  @date 03.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_NORMAL_H_
#define LVR2_GEOMETRY_NORMAL_H_

#include <ostream>

// Eigen sometimes produces errors when compiled with CUDA. Disables
// all Eigen related function for CUDA code (which is currently fine).
#ifndef __NVCC__
#include <Eigen/Dense>
#endif

#include "lvr2/geometry/BaseVector.hpp"

namespace lvr2
{


/**
 * @brief A vector guaranteed to be normalized (length = 1).
 *
 * If you have an object of type `Normal`, you can be sure that it always has
 * the length 1. The easiest way to create a `Normal` is to use the method
 * `Vector::normalized()`.
 */
template <typename CoordType>
struct Normal : public BaseVector<CoordType>
{
    Normal() { this->x = 0; this->y = 1; this->z = 0;}


    // ^ Private inheritance to restrict modifying access to the vector's data
    // in order to prevent modifications that would result in a non-normalized
    // vector.

    /**
     * @brief Creates a normal vector from the underlying vector representation.
     *
     * @param base This vector must not be the null-vector, else the behavior
     *             is undefined.
     */
    explicit Normal(BaseVector<CoordType> base);

    /**
     * @brief Initializes the normal with the given coordinates
     *
     * Note that the given coordinates must not form the null-vector, else the
     * behavior is undefined.
     */
    Normal(
        CoordType x,
        CoordType y,
        CoordType z
    );


    // Since the fields x, y and z can't be access directly anymore (else the
    // user could invalidate this *normal*), we provide getter methods.
    CoordType getX() const
    {
        return this->x;
    }
    
    CoordType getY() const
    {
        return this->y;
    }
    
    CoordType getZ() const
    {
        return this->z;
    }

    /**
     * @brief Returns the average of all normals in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be normals. It has to contain at least one element.
     */
    template<typename CollectionT>
    static Normal<CoordType>& average(const CollectionT& normals);

    /// Allows to assign Vectors to normals. Vector data will be copied
    /// and normalized.
    template<typename T>
    Normal<CoordType>& operator=(const T& other);

    template<typename T>
    Normal<CoordType> operator+(const T& other) const;

    template<typename T>
    Normal<CoordType> operator-(const T& other) const;
    
    Normal<CoordType> operator-() const;

// Eigen sometimes produces errors when compiled with CUDA. Disables
// all Eigen related function for CUDA code (which is currently fine).
#ifndef __NVCC__
    // Friend declaration for Eigen multiplication
    template<typename T, typename S>
    friend Normal<T> operator*(const Eigen::Matrix<S, 4, 4>& mat, const Normal<T>& normal);
#endif // ifndef __NVCC__

};

template<typename CoordType>
inline std::ostream& operator<<(std::ostream& os, const Normal<CoordType>& n)
{
    os << "Normal[" << n.getX() << ", " << n.getY() << ", " << n.getZ() << "]";
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
inline Normal<CoordType> operator*(const Eigen::Matrix<Scalar, 4, 4>& mat, const Normal<CoordType>& normal)
{
    // TODO: CHECK IF THIS IS CORRECT
    CoordType x = mat(0, 0) * normal.x + mat(1, 0) * normal.y + mat(2, 0) * normal.z;
    CoordType y = mat(0, 1) * normal.x + mat(1, 1) * normal.y + mat(2, 1) * normal.z;
    CoordType z = mat(0, 2) * normal.x + mat(1, 2) * normal.y + mat(2, 2) * normal.z;
    return Normal<CoordType>(x,y,z);
}
#endif // ifndef __NVCC__

} // namespace lvr2

#include "lvr2/geometry/Normal.tcc"

#endif /* LVR2_GEOMETRY_NORMAL_H_ */

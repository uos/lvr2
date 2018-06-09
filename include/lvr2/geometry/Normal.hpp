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
 * Normal.hpp
 *
 *  @date 03.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_NORMAL_H_
#define LVR2_GEOMETRY_NORMAL_H_

#include <ostream>

namespace lvr2
{

// Forward declarations
template <typename> struct Vector;


/**
 * @brief A vector guaranteed to be normalized (length = 1).
 *
 * If you have an object of type `Normal`, you can be sure that it always has
 * the length 1. The easiest way to create a `Normal` is to use the method
 * `Vector::normalized()`.
 */
template <typename BaseVecT>
struct Normal : private Vector<BaseVecT>
{
    // ^ Private inheritance to restrict modifying access to the vector's data
    // in order to prevent modifications that would result in a non-normalized
    // vector.

    /**
     * @brief Creates a normal vector from the underlying vector representation.
     *
     * @param base This vector must not be the null-vector, else the behavior
     *             is undefined.
     */
    explicit Normal(BaseVecT base);

    /**
     * @brief Creates a normal vector from a vector.
     *
     * Also see: `Vector::normalized()` which is easier to use in many
     * situations.
     *
     * @param base This vector must not be the null-vector, else the behavior
     *             is undefined.
     */
    explicit Normal(Vector<BaseVecT> vec);

    /**
     * @brief Initializes the normal with the given coordinates
     *
     * Note that the given coordinates must not form the null-vector, else the
     * behavior is undefined.
     */
    Normal(
        typename BaseVecT::CoordType x,
        typename BaseVecT::CoordType y,
        typename BaseVecT::CoordType z
    );


    // Since the fields x, y and z can't be access directly anymore (else the
    // user could invalidate this *normal*), we provide getter methods.
    typename BaseVecT::CoordType getX() const
    {
        return this->x;
    }
    typename BaseVecT::CoordType getY() const
    {
        return this->y;
    }
    typename BaseVecT::CoordType getZ() const
    {
        return this->z;
    }

    /**
     * @brief Returns the average of all normals in the given collection.
     *
     * The collection need to work with a range-based for-loop and its elements
     * need to be `Normal<BaseVecT>`. It has to contain at least one element.
     */
    template<typename CollectionT>
    static Normal<BaseVecT> average(const CollectionT& normals);

    // Cast normal to vector
    Vector<BaseVecT> asVector() const;

    // Publicly re-export methods that do not modify the vector and thus are
    // safe to use.
    using Vector<BaseVecT>::length;
    using Vector<BaseVecT>::length2;
    using Vector<BaseVecT>::cross;
    using Vector<BaseVecT>::dot;
    using Vector<BaseVecT>::operator==;
    using Vector<BaseVecT>::operator!=;
    using Vector<BaseVecT>::operator+;
    using Vector<BaseVecT>::operator-;

    // While already private, we delete these functions as they don't make
    // sense.
    BaseVecT& operator*=(const typename BaseVecT::CoordType &scale) = delete;
    BaseVecT& operator/=(const typename BaseVecT::CoordType &scale) = delete;

    Normal<BaseVecT> operator-() const;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const Normal<BaseVecT>& n)
{
    os << "Normal[" << n.getX() << ", " << n.getY() << ", " << n.getZ() << "]";
    return os;
}

} // namespace lvr2

#include <lvr2/geometry/Normal.tcc>

#endif /* LVR2_GEOMETRY_NORMAL_H_ */

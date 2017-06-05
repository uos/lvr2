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
 * BaseVector.tcc
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <cmath>

namespace lvr2
{

// ========================================================================
// === Named operations
// ========================================================================

template <typename CoordT>
CoordT BaseVector<CoordT>::length() const
{
    return sqrt(x * x + y * y + z * z);
}

template <typename CoordT>
CoordT BaseVector<CoordT>::length2() const
{
    return x * x + y * y + z * z;
}

template <typename CoordT>
CoordT BaseVector<CoordT>::distance(const BaseVector &other) const
{
    return (*this - other).length();
}

template <typename CoordT>
CoordT BaseVector<CoordT>::distance2(const BaseVector &other) const
{
    return (*this - other).length2();
}

template <typename CoordT>
BaseVector<CoordT> BaseVector<CoordT>::cross(const BaseVector &other) const
{
    auto tx = y * other.z - z * other.y;
    auto ty = z * other.x - x * other.z;
    auto tz = x * other.y - y * other.x;

    return BaseVector<CoordT>(tx, ty, tz);
}

template <typename CoordT>
CoordT BaseVector<CoordT>::dot(const BaseVector &other) const
{
    return x * other.x + y * other.y + z * other.z;
}


// ========================================================================
// === Operator overloads
// ========================================================================

template <typename CoordT>
BaseVector<CoordT> BaseVector<CoordT>::operator*(const CoordT &scale) const
{
    return BaseVector(*this) *= scale;
}

template <typename CoordT>
BaseVector<CoordT> BaseVector<CoordT>::operator/(const CoordT &scale) const
{
    return BaseVector(*this) /= scale;
}

template <typename CoordT>
BaseVector<CoordT>& BaseVector<CoordT>::operator*=(const CoordT &scale)
{
    x *= scale;
    y *= scale;
    z *= scale;

    return *this;
}

template <typename CoordT>
BaseVector<CoordT>& BaseVector<CoordT>::operator/=(const CoordT &scale)
{
    x /= scale;
    y /= scale;
    z /= scale;

    return *this;
}

template <typename CoordT>
BaseVector<CoordT> BaseVector<CoordT>::operator+(const BaseVector &other) const
{
    return BaseVector(*this) += other;
}

template <typename CoordT>
BaseVector<CoordT> BaseVector<CoordT>::operator-(const BaseVector &other) const
{
    return BaseVector(*this) -= other;
}

template <typename CoordT>
BaseVector<CoordT>& BaseVector<CoordT>::operator+=(const BaseVector<CoordT> &other)
{
    x += other.x;
    y += other.y;
    z += other.z;

    return *this;
}

template <typename CoordT>
BaseVector<CoordT>& BaseVector<CoordT>::operator-=(const BaseVector<CoordT> &other)
{
    x -= other.x;
    y -= other.y;
    z -= other.z;

    return *this;
}

template <typename CoordT>
bool BaseVector<CoordT>::operator==(const BaseVector &other) const
{
    return x == other.x
        && y == other.y
        && z == other.z;
}
template <typename CoordT>
bool BaseVector<CoordT>::operator!=(const BaseVector &other) const
{
    return !(this == other);
}

} // namespace lvr

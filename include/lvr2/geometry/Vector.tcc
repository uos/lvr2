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
 * Vector.tcc
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <cassert>

namespace lvr2
{

template <typename BaseVecT>
Normal<BaseVecT> Vector<BaseVecT>::normalized() const
{
    return Normal<BaseVecT>(*this);
}

template <typename BaseVecT>
void Vector<BaseVecT>::normalize()
{
    // Check for invalid vector. This check can be disabled in release mode,
    // which will probably lead to +inf and -inf values. In the documentation
    // for this function we require the vector to not be the null-vector.
    assert(!(this->x == 0 && this->y == 0 && this->z == 0));

    auto len = this->length();
    this->x /= len;
    this->y /= len;
    this->z /= len;
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator+(const Vector<BaseVecT>& other) const
{
    return BaseVecT::operator+(other);
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator-(const Vector<BaseVecT>& other) const
{
    return BaseVecT::operator-(other);
}

template <typename BaseVecT>
Vector<BaseVecT>& Vector<BaseVecT>::operator+=(const Vector<BaseVecT>& other)
{
    return static_cast<Vector<BaseVecT>&>(BaseVecT::operator+=(other));
}

template <typename BaseVecT>
Vector<BaseVecT>& Vector<BaseVecT>::operator-=(const Vector<BaseVecT>& other)
{
    return static_cast<Vector<BaseVecT>&>(BaseVecT::operator-=(other));
}

template <typename BaseVecT>
Point<BaseVecT> Vector<BaseVecT>::operator+(const Point<BaseVecT>& other) const
{
    return BaseVecT::operator+(other);
}

template <typename BaseVecT>
Point<BaseVecT> Vector<BaseVecT>::operator-(const Point<BaseVecT>& other) const
{
    return BaseVecT::operator-(other);
}

} // namespace lvr2

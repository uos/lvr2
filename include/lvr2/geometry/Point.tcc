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
 * Point.tcc
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <functional>

namespace lvr2
{

template <typename BaseVecT>
typename BaseVecT::CoordType Point<BaseVecT>::distanceFrom(const Point<BaseVecT> &other) const
{
    return sqrt((this->x - other.x)*(this->x - other.x) + (this->y - other.y)*(this->y - other.y) + (this->z - other.z)*(this->z - other.z));
}

template <typename BaseVecT>
Point<BaseVecT> Point<BaseVecT>::operator+(const Vector<BaseVecT> &other) const
{
    return BaseVecT::operator+(other);
}

template <typename BaseVecT>
Point<BaseVecT> Point<BaseVecT>::operator-(const Vector<BaseVecT> &other) const
{
    return BaseVecT::operator-(other);
}

template <typename BaseVecT>
Point<BaseVecT>& Point<BaseVecT>::operator+=(const Vector<BaseVecT> &other)
{
    return static_cast<Point<BaseVecT>&>(BaseVecT::operator+=(other));
}

template <typename BaseVecT>
Point<BaseVecT>& Point<BaseVecT>::operator-=(const Vector<BaseVecT> &other)
{
    return static_cast<Point<BaseVecT>&>(BaseVecT::operator-=(other));
}

template <typename BaseVecT>
Vector<BaseVecT> Point<BaseVecT>::operator-(const Point<BaseVecT> &other) const
{
    return BaseVecT::operator-(other);
}

template <typename BaseVecT>
Vector<BaseVecT> Point<BaseVecT>::asVector() const
{
    return static_cast<BaseVecT>(*this);
}

} // namespace lvr2

namespace std
{

template<typename BaseVecT>
struct hash<lvr2::Point<BaseVecT>> {
    size_t operator()(const lvr2::Point<BaseVecT>& point) const
    {
        return std::hash<typename BaseVecT::CoordType>()(point.x)
               ^ std::hash<typename BaseVecT::CoordType>()(point.y)
               ^ std::hash<typename BaseVecT::CoordType>()(point.z);
    }
};

} // namespace std

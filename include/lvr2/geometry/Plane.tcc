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
 * Plane.tcc
 *
 *  @date 17.07.2017
 *  @author @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include "Plane.hpp"

namespace lvr2
{

template<typename BaseVecT>
float Plane<BaseVecT>::distance(const Vector<BaseVecT>& other) const
{
    // Distance betweeen plane and query point (calculated by hesse normal form)
    // Credits: https://oberprima.com/mathematik/abstand-berechnen/
    return (other - this->pos).dot(this->normal);
}

template<typename BaseVecT>
Vector<BaseVecT> Plane<BaseVecT>::project(const Vector<BaseVecT>& other) const
{
    // Credits to: https://stackoverflow.com/questions/9605556/#answer-41897378
    return other - (this->normal * (this->normal.dot(other - this->pos)));
}

template<typename BaseVecT>
Line<BaseVecT> Plane<BaseVecT>::intersect(const Plane <BaseVecT>& other) const {
    float d1 = normal.dot(pos);
    float d2 = other.normal.dot(other.pos);
    auto direction = normal.cross(other.normal);

    Line<BaseVecT> intersection;
    intersection.normal = direction.normalized();
    intersection.pos = (other.normal * d1 - normal * d2).cross(direction)
                       * (1 / (direction.dot(direction)));

    return intersection;
}

} // namespace lvr2

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
 * Plane.tcc
 *
 *  @date 17.07.2017
 *  @author @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include "Plane.hpp"

namespace lvr2
{

template<typename BaseVecT>
float Plane<BaseVecT>::distance(const BaseVecT& other) const
{
    // Distance betweeen plane and query point (calculated by hesse normal form)
    // Credits: https://oberprima.com/mathematik/abstand-berechnen/
    return (other - this->pos).dot(this->normal);
}

template<typename BaseVecT>
BaseVecT Plane<BaseVecT>::project(const BaseVecT& other) const
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

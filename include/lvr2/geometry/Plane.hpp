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
 * Plane.hpp
 *
 *  @date 14.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_GEOMETRY_PLANE_H_
#define LVR2_GEOMETRY_PLANE_H_

#include "Normal.hpp"
#include "Line.hpp"

namespace lvr2
{

/**
 * @brief A plane.
 *
 * A plane represented by a normal and a position vector.
 */
template <typename BaseVecT>
struct Plane
{
    Plane() : normal(0, 0, 1) {}

    Normal<typename BaseVecT::CoordType> normal;
    BaseVecT pos;

    /// Projects the given point onto the plane and returns the projection point.
    BaseVecT project(const BaseVecT& other) const;

    /**
     * @brief Calculates the distance between the plane and the given point.
     * @return This can be < 0, == 0 or > 0 the cases mean:
     *         < 0: The point lies between the plane and the origin
     *         == 0: The point lies in the plane
     *         > 0: The point lies behind the plane, oberserved from the origin
     */
    float distance(const BaseVecT& other) const;

    /// Calculates the intersection between this and other
    Line<BaseVecT> intersect(const Plane<BaseVecT>& other) const;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const Plane<BaseVecT>& p)
{
    os << "Plane[" << p.normal << ", " << p.pos << "]";
    return os;
}

} // namespace lvr2

#include "lvr2/geometry/Plane.tcc"

#endif /* LVR2_GEOMETRY_PLANE_H_ */

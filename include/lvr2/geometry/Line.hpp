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
 * Line.hpp
 *
 *  @date 06.10.2017
 *  @author Christian Swan cswan@uos.de
 */

#ifndef LVR2_GEOMETRY_LINE_H_
#define LVR2_GEOMETRY_LINE_H_

#include "Normal.hpp"
#include "Point.hpp"

namespace lvr2
{

/**
 * @brief A Line.
 *
 * A line represented by a normal and a position vector.
 */
template <typename BaseVecT>
struct Line
{
    Line() : normal(0, 0, 1) {}

    Normal<BaseVecT> normal;
    Point<BaseVecT> pos;

    /// Projects the given point onto the line and returns the projection point.
    Point<BaseVecT> project(const Point<BaseVecT>& other) const;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const Line<BaseVecT>& l)
{
    os << "Line[" << l.normal << ", " << l.pos << "]";
    return os;
}

} // namespace lvr2

#include <lvr2/geometry/Line.tcc>

#endif /* LVR2_GEOMETRY_LINE_H_ */

/* Copyright (C) 2016 Uni Osnabrück
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
 * VertexTraits.hpp
 *
 *  Created on: Dec 20, 2016
 *      Author: Thomas Wiemann (twiemann@uos.de)
 */

#ifndef INCLUDE_LVR_GEOMETRY_VERTEXTRAITS_HPP_
#define INCLUDE_LVR_GEOMETRY_VERTEXTRAITS_HPP_

#include "ColorVertex.hpp"

namespace lvr
{

template<typename VertexT>
struct VertexTraits
{
    static const bool HasColor = false;

    /// Wrapper function to set colors. Does nothing if colors are not supported.
    static void setColor(VertexT& v, unsigned char r, unsigned char g, unsigned char b) {}
};

template<>
struct VertexTraits<ColorVertex<float, unsigned char> >
{
    static const bool HasColor = true;

    /// Wrapper function set the rgb fields of an vertex that supports colors
    static void setColor(ColorVertex<float, unsigned char> & v, unsigned char r, unsigned char g, unsigned char b)
    {
        v.r = r;
        v.g = g;
        v.b = b;
    }
};

}

#endif /* INCLUDE_LVR_GEOMETRY_VERTEXTRAITS_HPP_ */
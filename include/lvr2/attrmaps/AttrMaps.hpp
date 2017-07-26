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
 * AttributeMap.hpp
 *
 *  @date 26.07.2017
 */

#ifndef LVR2_ATTRMAPS_ATTRMAPS_H_
#define LVR2_ATTRMAPS_ATTRMAPS_H_

#include <lvr2/attrmaps/AttributeMap.hpp>
#include <lvr2/attrmaps/VectorMap.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

// ===========================================================================
// Useful typedefs
// ===========================================================================

template<typename HandleT, typename ValueT>
using DenseAttrMap = VectorMap<HandleT, ValueT>;


template<typename ValueT>
using EdgeMap = AttributeMap<EdgeHandle, ValueT>;

template<typename ValueT>
using FaceMap = AttributeMap<FaceHandle, ValueT>;

template<typename ValueT>
using VertexMap = AttributeMap<VertexHandle, ValueT>;

template<typename ValueT>
using ClusterMap = VectorMap<ClusterHandle, ValueT>;

template<typename ValueT>
using DenseEdgeMap = DenseAttrMap<EdgeHandle, ValueT>;

template<typename ValueT>
using DenseFaceMap = DenseAttrMap<FaceHandle, ValueT>;

template<typename ValueT>
using DenseVertexMap = DenseAttrMap<VertexHandle, ValueT>;


} // namespace lvr2

#endif /* LVR2_ATTRMAPS_ATTRMAPS_H_ */

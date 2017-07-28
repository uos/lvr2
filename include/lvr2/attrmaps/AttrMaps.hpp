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
#include <lvr2/attrmaps/HashMap.hpp>
#include <lvr2/attrmaps/ListMap.hpp>
#include <lvr2/attrmaps/VectorMap.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

/*
 * This file defines many useful type aliases for implementors of the
 * `AttributeMap` interface.
 *
 * Choosing the correct implementation can have a huge impact on performance
 * of your algorithm. There are three main implementations, where each is
 * useful in a specific situation -- it mainly depends on the number of values
 * in the map.
 *
 * - DenseAttrMap: if there is a value associated with almost all handles
 * - SparseAttrMap: if the number of values is significantly less than the
 *                  number of handles
 * - TinyAttrMap: if there is only a very small number of values (say... 7).
 *
 * In some algorithms you will associate a value with *every* handle, e.g. a
 * `VertexMap<bool> visited`. In this situation, it's useful to use the
 * DenseAttrMap: it will be faster than the other two implementations. In other
 * situations, however, you won't associate a value with most handles. Here,
 * a DenseAttrMap would be a bad idea, because it has a memory requirement of
 * O(biggest_handle_idx). This means, a lot of space would be wasted. Thus,
 * rather use SparseAttrMap in that case. Each lookup will be a bit slower, but
 * we won't waste memory. Lastely, the `TinyAttrMap` is useful if you know you
 * will have very few values.
 *
 * It's useful to look at HOW these implementations work under the hood to
 * understand the runtime and memory overhead of each:
 *
 * - DenseAttrMap: uses an array (VectorMap/std::vector)
 * - SparseAttrMap: uses a hash map (HashMap/std::unordered_map)
 * - TinyAttrMap: uses an unsorted list of key-value pairs
 *
 *
 * Additionally, there are specific type aliases for the most common uses,
 * like `FaceMap = AttributeMap<FaceHandle, T>`. You should use those when you
 * can.
 */

// ---------------------------------------------------------------------------
// Generic aliases
template<typename HandleT, typename ValueT> using DenseAttrMap  = VectorMap<HandleT, ValueT>;
template<typename HandleT, typename ValueT> using SparseAttrMap = HashMap<HandleT, ValueT>;
template<typename HandleT, typename ValueT> using TinyAttrMap   = ListMap<HandleT, ValueT>;

// ---------------------------------------------------------------------------
// Handle-specific aliases
template<typename ValueT> using ClusterMap  = AttributeMap<ClusterHandle, ValueT>;
template<typename ValueT> using EdgeMap     = AttributeMap<EdgeHandle, ValueT>;
template<typename ValueT> using FaceMap     = AttributeMap<FaceHandle, ValueT>;
template<typename ValueT> using VertexMap   = AttributeMap<VertexHandle, ValueT>;

 // ---------------------------------------------------------------------------
 // Handle- and implementation-specific aliases
template<typename ValueT> using DenseClusterMap     = DenseAttrMap<ClusterHandle, ValueT>;
template<typename ValueT> using DenseEdgeMap        = DenseAttrMap<EdgeHandle, ValueT>;
template<typename ValueT> using DenseFaceMap        = DenseAttrMap<FaceHandle, ValueT>;
template<typename ValueT> using DenseVertexMap      = DenseAttrMap<VertexHandle, ValueT>;

template<typename ValueT> using SparseClusterMap    = SparseAttrMap<ClusterHandle, ValueT>;
template<typename ValueT> using SparseEdgeMap       = SparseAttrMap<EdgeHandle, ValueT>;
template<typename ValueT> using SparseFaceMap       = SparseAttrMap<FaceHandle, ValueT>;
template<typename ValueT> using SparseVertexMap     = SparseAttrMap<VertexHandle, ValueT>;

 template<typename ValueT> using TinyClusterMap     = TinyAttrMap<ClusterHandle, ValueT>;
 template<typename ValueT> using TinyEdgeMap        = TinyAttrMap<EdgeHandle, ValueT>;
 template<typename ValueT> using TinyFaceMap        = TinyAttrMap<FaceHandle, ValueT>;
 template<typename ValueT> using TinyVertexMap      = TinyAttrMap<VertexHandle, ValueT>;

} // namespace lvr2

#endif /* LVR2_ATTRMAPS_ATTRMAPS_H_ */

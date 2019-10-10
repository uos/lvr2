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
 * AttributeMap.hpp
 *
 *  @date 26.07.2017
 */

#ifndef LVR2_ATTRMAPS_ATTRMAPS_H_
#define LVR2_ATTRMAPS_ATTRMAPS_H_

#include "lvr2/attrmaps/AttributeMap.hpp"
#include "lvr2/attrmaps/HashMap.hpp"
#include "lvr2/attrmaps/ListMap.hpp"
#include "lvr2/attrmaps/VectorMap.hpp"
#include "lvr2/geometry/Handles.hpp"

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

 template<typename ValueT> using DenseVertexMapOptional = boost::optional<DenseVertexMap<ValueT>>; 
} // namespace lvr2

#endif /* LVR2_ATTRMAPS_ATTRMAPS_H_ */

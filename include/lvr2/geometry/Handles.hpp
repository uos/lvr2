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
 * Handles.hpp
 *
 *  @date 15.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_HANDLES_H_
#define LVR2_GEOMETRY_HANDLES_H_

#include <cstdint>
#include <functional>

#include "lvr2/util/BaseHandle.hpp"

namespace lvr2
{

/**
 * @brief Datatype used as index for each vertex, face and edge.
 *
 * This index is used within {Edge, Face, Vertex}-Handles. Since those
 * handles are also used within each {Edge, Face, Vertex} reducing the
 * size of this type can greatly decrease memory usage, which in
 * turn might increase performance due to cache locality.
 *
 * When we assume a basic half-edge structure, we have to deal with the
 * following struct sizes:
 * - Edge: 4 handles
 * - Face: 1 handle + 1 vector
 * - Vertex: 1 handle + 1 vector
 *
 * Assuming the most common case of `float` vectors, this results in the
 * following sizes (in bytes):
 * - 16 bit handles: Edge (8), Face (14), Vertex (14)
 * - 32 bit handles: Edge (16), Face (16), Vertex (16)
 * - 64 bit handles: Edge (32), Face (20), Vertex (20)
 *
 * Using another approximation of the number of faces, edges and vertices
 * in a triangle-mesh described at [1], we can calculate how much RAM we
 * would need in order to run out of handles. The approximation: for each
 * vertex, we have three edges and two faces. The de-facto cost per vertex
 * can be calculated from that resulting in
 *
 * - 16 bit handles: 14 + 2*14 + 3*8 = 66 bytes/vertex = 22 bytes/edge
 *   ==> 22 * 2^16 = 1.4 MiB RAM necessary to exhaust handle space
 *
 * - 32 bit handles: 16 + 2*16 + 3*16 = 96 bytes/vertex = 32 bytes/edge
 *   ==> 32 * 2^32 = 137 GiB RAM necessary to exhaust handle space
 *
 * - 64 bit handles: 20 + 2*20 + 3*32 = 156 bytes/vertex = 52 bytes/edge
 *   ==> 52 * 2^64 = 1.1 ZiB RAM necessary to exhaust handle space
 *       (it's called zetta or zebi and is ≈ 1 million tera bytes)
 *   ==> Note: funnily enough, the estimated disk (not RAM!) capacity of
 *       the whole  world (around 2015) comes very close to this number.
 *
 *
 * Also note that this accounts for the mesh only and ignores all other
 * data that might need to be stored in RAM. So you will need even more
 * RAM.
 *
 * From this, I think, we can safely conclude: 16 bit handles are way too
 * small; 32 bit handles are probably fine for the next few years, even
 * when working on a medium-sized cluster and 64 bit handles will be fine
 * until after the singularity. And by then, I probably don't care anymore.
 *
 * [1]: https://math.stackexchange.com/q/425968/340615
 */
using Index = uint32_t;

// Note on strongly typed handles:
//
// You might ask: Why do we need that many classes for handles? Wouldn't one
// be enough? Or you go even further: if every handle is just a simple integer,
// why not store the integer directly. Well, it all comes down to "strong
// typing".
//
// Type systems are the main way for compilers to notice that a program is
// faulty. While compiler errors are just annoying in the first few years of
// learning how to program, they become very useful later on. When writing
// software, humans will make mistakes -- that's just a fact. The question is
// WHEN we want to notice those mistakes. There are a few possibilities here:
//
// - while compiling
// - while executing unit tests
// - in production
//
// No one wants to notice bugs when already running software in production. Thus
// we want to notice our mistakes earlier. Since this whole library clearly
// doesn't care about unit tests, mistakes can only be noticed either at
// compile time or when the developer executes the program.
//
// Well, now the fun parts. When you use a language with a sufficiently
// powerful type system (as C++) and if you are correctly using this type
// system, you can avoid many huge classes of bugs! The compiler will tell you
// right away, when you made a mistake.
//
// So with these strongly typed handles, you cannot falsely assign an EdgeHandle
// to a FaceHandle -- it will result in a compiler error. If you were using
// simple integers, the compiler wouldn't notice and you would have to track
// down the bug manually. Not so great.
//
// Apart from that: it makes reading code so much easier, as you know exactly
// what a specific parameter is for.

/// Handle to access edges of the mesh.
class EdgeHandle : public BaseHandle<Index>
{
    using BaseHandle<Index>::BaseHandle;
};

/// Handle to access faces of the mesh.
class FaceHandle : public BaseHandle<Index>
{
    using BaseHandle<Index>::BaseHandle;
};

/// Handle to access vertices of the mesh.
class VertexHandle : public BaseHandle<Index>
{
    using BaseHandle<Index>::BaseHandle;
};

/// Handle to access Cluster of the ClusterBiMap.
class ClusterHandle : public BaseHandle<Index>
{
    using BaseHandle<Index>::BaseHandle;
};

/// Handle to access textures of the mesh
class TextureHandle : public BaseHandle<Index>
{
    using BaseHandle<Index>::BaseHandle;
};

/// Semantically equivalent to `boost::optional<EdgeHandle>`
class OptionalEdgeHandle : public BaseOptionalHandle<Index, EdgeHandle>
{
    using BaseOptionalHandle<Index, EdgeHandle>::BaseOptionalHandle;
};

/// Semantically equivalent to `boost::optional<FaceHandle>`
class OptionalFaceHandle : public BaseOptionalHandle<Index, FaceHandle>
{
    using BaseOptionalHandle<Index, FaceHandle>::BaseOptionalHandle;
};

/// Semantically equivalent to `boost::optional<VertexHandle>`
class OptionalVertexHandle : public BaseOptionalHandle<Index, VertexHandle>
{
    using BaseOptionalHandle<Index, VertexHandle>::BaseOptionalHandle;
};

/// Semantically equivalent to `boost::optional<ClusterHandle>`
class OptionalClusterHandle : public BaseOptionalHandle<Index, ClusterHandle>
{
    using BaseOptionalHandle<Index, ClusterHandle>::BaseOptionalHandle;
};

inline std::ostream& operator<<(std::ostream& os, const EdgeHandle& h)
{
    os << "E" << h.idx();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const FaceHandle& h)
{
    os << "F" << h.idx();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const VertexHandle& h)
{
    os << "V" << h.idx();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ClusterHandle& h)
{
    os << "C" << h.idx();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OptionalEdgeHandle& h)
{
    if (h)
    {
        os << "E" << h.unwrap().idx();
    }
    else
    {
        os << "E⊥";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OptionalFaceHandle& h)
{
    if (h)
    {
        os << "F" << h.unwrap().idx();
    }
    else
    {
        os << "F⊥";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OptionalVertexHandle& h)
{
    if (h)
    {
        os << "V" << h.unwrap().idx();
    }
    else
    {
        os << "V⊥";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OptionalClusterHandle& h)
{
    if (h)
    {
        os << "C" << h.unwrap().idx();
    }
    else
    {
        os << "C⊥";
    }
    return os;
}

} // namespace lvr2

namespace std
{

template<>
struct hash<lvr2::EdgeHandle> {
    size_t operator()(const lvr2::EdgeHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

template<>
struct hash<lvr2::FaceHandle> {
    size_t operator()(const lvr2::FaceHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

template<>
struct hash<lvr2::VertexHandle> {
    size_t operator()(const lvr2::VertexHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

template<>
struct less<lvr2::VertexHandle> {
    bool operator()(const lvr2::VertexHandle& l, const lvr2::VertexHandle& r) const
    {
        return std::less<lvr2::Index>()(l.idx(), r.idx());
    }
};

template<>
struct hash<lvr2::ClusterHandle> {
    size_t operator()(const lvr2::ClusterHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

template<>
struct hash<lvr2::TextureHandle> {
    size_t operator()(const lvr2::TextureHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

} // namespace std

#endif /* LVR2_GEOMETRY_HANDLES_H_ */

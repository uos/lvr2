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
 * HalfEdge.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_HALFEDGE_H_
#define LVR2_GEOMETRY_HALFEDGE_H_

#include <utility>

#include "BaseMesh.hpp"
#include "Handles.hpp"

namespace lvr2
{

// We need a specific handle for half edges. The `BaseMesh` interface talks
// about simple (full) edges. To avoid confusion, this HalfEdgeHandle is used
// internally in the HEM. The edge handles given out by the HEM implementation
// are handles to the half edge with the lower index.

/// Handle to access half edges of the HEM.
class HalfEdgeHandle : public BaseHandle<Index>
{
public:
    using BaseHandle<Index>::BaseHandle;

    static HalfEdgeHandle oneHalfOf(EdgeHandle eH)
    {
        // The index of an edge handle is always the index of the handle of
        // one of its half edges
        return HalfEdgeHandle(eH.idx());
    }
};

/// Semantically equivalent to `boost::optional<HalfEdgeHandle>`
class OptionalHalfEdgeHandle : public BaseOptionalHandle<Index, HalfEdgeHandle>
{
public:
    using BaseOptionalHandle<Index, HalfEdgeHandle>::BaseOptionalHandle;
    OptionalHalfEdgeHandle() : BaseOptionalHandle() {}
    OptionalHalfEdgeHandle(EdgeHandle eH) : OptionalHalfEdgeHandle(eH.idx()) {}
};

struct HalfEdge
{
    /// The face this edge belongs to (or none, if this edge lies on the
    /// boundary).
    OptionalFaceHandle face;

    /// The vertex this edge points to.
    VertexHandle target;

    /// The next edge of the face, ordered counter-clockwise. Viewed a different
    /// way: it's the next edge when walking clockwise around the source
    /// vertex.
    HalfEdgeHandle next;

    /// The twin edge.
    HalfEdgeHandle twin;

private:
    /**
     * @brief Initializes all fields with dummy values (unsafe, thus private).
     */
    HalfEdge() : target(0), next(0), twin(0) {}

    /// Several methods of HEM need to invoke the unsafe ctor.
    template <typename BaseVecT>
    friend class HalfEdgeMesh;
};


inline std::ostream& operator<<(std::ostream& os, const HalfEdgeHandle& h)
{
    os << "HE" << h.idx();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OptionalHalfEdgeHandle& h)
{
    if (h)
    {
        os << "HE" << h.unwrap().idx();
    }
    else
    {
        os << "HE⊥";
    }
    return os;
}

} // namespace lvr2

namespace std
{

template<>
struct hash<lvr2::HalfEdgeHandle> {
    size_t operator()(const lvr2::HalfEdgeHandle& h) const
    {
        return std::hash<lvr2::Index>()(h.idx());
    }
};

} // namespace std

#endif /* LVR2_GEOMETRY_HALFEDGE_H_ */

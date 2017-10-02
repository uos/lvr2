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
// are handle to the half edge with the lower index. Luckily a half edge pair
// always sits next to each other in the vector of all half edges. The half
// edge with lower index always has an even index.

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

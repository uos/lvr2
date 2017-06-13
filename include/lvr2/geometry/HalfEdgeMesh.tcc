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
 * HalfEdgeMesh.tcc
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <utility>
#include <array>

namespace lvr2
{

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::VertexHandle
    HalfEdgeMesh<BaseVecT>::addVertex(Point<BaseVecT> pos)
{
    Vertex v;
    v.pos = pos;
    m_vertices.push_back(v);
    return VertexHandle(static_cast<Index>(m_vertices.size() - 1));
}

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::FaceHandle
    HalfEdgeMesh<BaseVecT>::addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H)
{
    using std::array;
    using std::pair;
    using std::make_pair;

    // The order here matters! This way, only the next-handles of the last
    // edge might be broken.
    auto e1H = findOrCreateEdgeBetween(v1H, v2H);
    auto e2H = findOrCreateEdgeBetween(v2H, v3H);
    auto e3H = findOrCreateEdgeBetween(v3H, v1H);

    // Create face
    FaceHandle newFaceH = m_faces.size();

    // Calc normal
    // TODO: move to seperate method
    auto v1 = getV(v1H).pos;
    auto v2 = getV(v1H).pos;
    auto v3 = getV(v1H).pos;
    auto normal = (v1 - v2).cross(v1 - v3);

    Face f(e1H, Normal<BaseVecT>(normal));
    m_faces.push_back(f);

    // Set face
    getE(e1H).face = newFaceH;
    getE(e2H).face = newFaceH;
    getE(e3H).face = newFaceH;

    // Fix `next` handles that might be invalid after adding all edges
    // independent from another.
    if (getE(e3H).next != e1H)
    {
        // We know that after we added the edges, `v1` has at least one
        // outgoing edge.
        auto boundaryEdgeH = boundaryEdgeOf(v1H).unwrap();

        auto prevNextH = getE(e3H).next;
        getE(e3H).next = e1H;
        getE(getE(e1H).twin).next = prevNextH;
        getE(boundaryEdgeH).next = getE(e3H).twin;
    }

    // array<EdgeHandle, 3> innerEdges;
    // array<pair<EdgeHandle, EdgeHandle>, 3> edgeEndpoints = {
    //     make_pair(v1, v2),
    //     make_pair(v2, v3),
    //     make_pair(v3, v1)
    // };

    // for (auto i : {0, 1, 2})
    // {
    //     auto endpoints = edgeEndpoints[i];
    //     auto edgeH = edgeBetween(endpoints.first, endpoints.second);
    //     if (!edgeH)
    //     {
    //         auto pair = addEdgePair();
    //         getE(pair.first).target = endpoints.first;
    //         getE(pair.second).target = endpoints.second;
    //         edgeH = pair.second;
    //     }
    //     innerEdges[i] = edgeH;
    // }

    for (auto i : {0, 1, 2})
    {
        // auto innerEdgeH = innerEdges[i];
        // auto nextInnerEdgeH = innerEdges[(i + 1) % 3];
        // auto vertexH = getE(innerEdgeH).target;

        // getE(innerEdgeH).next = nextInnerEdgeH;

        // We need to add two edges to the vertex such that you can iterate
        // over all edges leaving or entering the vertex using the `next`
        // handles. This is possible in a non-broken HEM.
        //
        // Let's take a look at this example (I'm sorry for ugly ascii art):
        //
        //        x  y      z  w        |  where:
        //         ^ \      ^ /         |  x = innerEdge.twin
        //          \ \    / /          |  y = innerEdge
        //           \ v  / v           |  z = nextInnerEdge
        //              ()              |  w = nextInnerEdge.twin
        //           ^ /  ^ \           |
        //          / /    \ \          |  And a, b, c and d already existed
        //         / v      \ v         |  before.
        //        a  b      c  d        |
        //
        //
        // Note that (before we change anything) these statements are true:
        // - a.next == d
        // - c.next == b
        //
        // What we want to achieve is:
        // - a.next = x
        // - y.next = z  (this was already done above!)
        // - w.next = d
        // - c.next = b  (this is exactly like before)
        //
        // Of course, this example is a special case: there might be more than
        // two edges connected to that one vertex. We need to insert our two
        // edges somewhere into the cycle around the vertex. Again, to make
        // this clear, the cycle means:
        //
        //     v.outgoing == v.outgoing.twin.next.twin.next ... .twin.next
        //
        // We have to pay attention to where we insert our two edges. We can't
        // do it between two edges belonging to one face. We can only change
        // `next` handles of edges that don't belong to a face!
        //
        // And in fact, here the case of non-manifold vertices can occur.
        // Maybe. I really don't know if this case can ever occur when using
        // marching cubes. It might not. So for now, we will assert that it
        // doesn't happen. But we need to...
        //
        // TODO: understand and fix this stuff!
        // auto aH = getV(vertexH).outgoing;
        // while (getE(aH).face)
        // {
        //     aH = getE(getE(aH).next).twin;

        //     // Test if we reached the start edge again
        //     if (aH == getV(vertexH).outgoing)
        //     {
        //         // In this case a non-manifold vertex would be created. We
        //         // could potentially easily fix it by duplicating the vertex
        //         // and treating both vertices as distinct ones. But for now,
        //         // we simply assert that this never happens.
        //         assert(false);
        //     }
        // }

        // auto dH = getE(aH).next;
        // getE(aH).next = getE(innerEdgeH).twin;
        // getE(nextInnerEdgeH).next = dH;
    }
}

// ========================================================================
// = Private helper methods
// ========================================================================

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Edge&
    HalfEdgeMesh<BaseVecT>::getE(EdgeHandle handle)
{
    return m_edges[handle.idx()];
}

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Face&
    HalfEdgeMesh<BaseVecT>::getF(FaceHandle handle)
{
    return m_faces[handle.idx()];
}

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Vertex&
    HalfEdgeMesh<BaseVecT>::getV(VertexHandle handle)
{
    return m_vertices[handle.idx()];
}

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::edgeBetween(VertexHandle fromH, VertexHandle toH)
{
    auto& from = getV(fromH);

    // Is there even a single outgoing edge?
    if (from.outgoing)
    {
        auto startH = from.outgoing.unwrap();

        auto edgeH = startH;
        do
        {
            auto& edge = getE(edgeH);
            if (edge.target == toH)
            {
                return edgeH;
            }
            edgeH = getE(edge.twin).next;
        } while(edgeH != startH);
    }

    // Return none.
    return OptionalEdgeHandle();
}

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::EdgeHandle
    HalfEdgeMesh<BaseVecT>::findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH)
{
    auto foundEdge = edgeBetween(fromH, toH);
    if (foundEdge)
    {
        return foundEdge.unwrap();
    }
    else
    {
        return addEdgePair(fromH, toH).first;
    }
}


template <typename BaseVecT>
pair<typename BaseMesh<BaseVecT>::EdgeHandle, typename BaseMesh<BaseVecT>::EdgeHandle>
    HalfEdgeMesh<BaseVecT>::addEdgePair(VertexHandle v1H, VertexHandle v2H)
{
    //
    // This method adds two new half edges, called "a" and "b".
    //
    //  +----+  --------(a)-------->  +----+
    //  | v1 |                        | v2 |
    //  +----+  <-------(b)---------  +----+
    //
    //
    //
    //
    //
    //        x  y      z  w        |
    //         ^ \      ^ /         |
    //          \ \    / /          |
    //           \ v  / v           |
    //             [v1]
    //                              |
    //           ^ /  ^ \           |
    //          / /    \ \          |  And a, b, c and d already existed
    //         / v      \ v         |  before.
    //        a  b      c  d        |
    //
    //


    // Create incomplete/broken edges and edge handles. By the end of this
    // method, they are less invalid.
    Edge a;
    Edge b;
    EdgeHandle aH(m_edges.size());
    EdgeHandle bH(m_edges.size() + 1);

    // Assign twins to each other
    a.twin = bH;
    b.twin = aH;

    // Assign half-edge targets
    a.target = v2H;
    b.target = v1H;

    auto fixNextHandles = [&](VertexHandle vH, EdgeHandle ingoingH, EdgeHandle outgoingH)
    {
        auto& v = getV(vH);
        if (v.outgoing)
        {
            // We already checked that `v` has an outgoing edge, so we can
            // unwrap.
            auto boundaryEdgeH = boundaryEdgeOf(vH).unwrap();

            // Visualization:
            //
            //
            //             ^  ...  /
            //  (prevNext)  \     /  (boundaryEdge)
            //               \   /
            //                \ v
            //                [v]
            //                ^ |
            //                | |
            //           (in) | | (out)
            //                | v
            //
            //
            auto prevNextH = getE(boundaryEdgeH).next;
            getE(boundaryEdgeH).next = outgoingH;
            getE(ingoingH).next = prevNextH;
        }
        else
        {
            // This is the easy case: the vertex was not connected before, so
            // we don't need to change any handles of other edges.
            //
            //         [v]
            //         ^ |
            //         | |
            //    (in) | | (out)
            //         | v
            //
            v.outgoing = outgoingH;
            getE(ingoingH).next = outgoingH;
        }
    };

    fixNextHandles(v1H, bH, aH);
    fixNextHandles(v2H, aH, bH);

    // Add edges to vector.
    m_edges.push_back(a);
    m_edges.push_back(b);

    return std::make_pair(aH, bH);
}


template <typename BaseVecT>
typename BaseMesh<BaseVecT>::OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::boundaryEdgeOf(VertexHandle vH)
{
    auto& v = getV(vH);
    if (v.outgoing)
    {
        const auto startEdgeH = getE(v.outgoing.unwrap()).twin;
        auto boundaryEdgeH = startEdgeH;
        while (getE(boundaryEdgeH).face)
        {
            boundaryEdgeH = getE(getE(boundaryEdgeH).next).twin;
            if (boundaryEdgeH == startEdgeH)
            {
                // In this case a non-manifold vertex would be created. We
                // could potentially easily fix it by duplicating the vertex
                // and treating both vertices as distinct ones. But for now,
                // we simply assert that this never happens.
                assert(false);
            }
        }
        return boundaryEdgeH;
    }
    return OptionalEdgeHandle();
}




} // namespace lvr

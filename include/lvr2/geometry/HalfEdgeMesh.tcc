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
#include <iostream>

using namespace std;

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
    cout << "##################################################" << endl;
    cout << "##### addFace(): " << v1H << " -> " << v2H << " -> " << v3H << endl;

    using std::array;
    using std::make_tuple;

    // Handles for the inner edges of the face. The edges represented by those
    // handles do not contain a valid `next` pointer yet.
    auto eInner1H = findOrCreateEdgeBetween(v1H, v2H);
    auto eInner2H = findOrCreateEdgeBetween(v2H, v3H);
    auto eInner3H = findOrCreateEdgeBetween(v3H, v1H);

    auto& eInner1 = getE(eInner1H);
    auto& eInner2 = getE(eInner2H);
    auto& eInner3 = getE(eInner3H);

    auto eOuter1H = eInner1.twin;
    auto eOuter2H = eInner2.twin;
    auto eOuter3H = eInner3.twin;

    auto& eOuter1 = getE(eOuter1H);
    auto& eOuter2 = getE(eOuter2H);
    auto& eOuter3 = getE(eOuter3H);

    auto v1 = getV(v1H);
    auto v2 = getV(v2H);
    auto v3 = getV(v3H);


    // Create face
    FaceHandle newFaceH = m_faces.size();

    // Calc normal
    // TODO: move to seperate method
    auto normal = (v1.pos - v2.pos).cross(v1.pos - v3.pos);

    Face f(eInner1H, Normal<BaseVecT>(normal));
    m_faces.push_back(f);

    // Set face
    getE(eInner1H).face = newFaceH;
    getE(eInner2H).face = newFaceH;
    getE(eInner3H).face = newFaceH;


    // Fix next handles and set outgoing handles if net set yet
    auto corners = {
        make_tuple(eOuter1H, v1H, eOuter3H),
        make_tuple(eOuter2H, v2H, eOuter1H),
        make_tuple(eOuter3H, v3H, eOuter2H)
    };

    for (auto corner : corners)
    {
        auto eInH  = get<0>(corner);
        auto vH    = get<1>(corner);
        auto eOutH = get<2>(corner);

        auto& eIn = getE(eInH);
        auto& v = getV(vH);
        auto& eOut = getE(eOutH);

        // --> Case (A): neither edge is part of a face (both edges are new)
        if (!eIn.face && !eOut.face)
        {
            // We need to handle the special case of `v` not having an
            // outgoing edge.
            if (v.outgoing)
            {
                // First we need to find two boundary edges adjacent to `v`. One
                // ending at `v`, called eEnd, and one starting at `v`, called
                // eStart. Note that `eEnd.next == eStart`.
                //
                // Since we already know that `v.outgoing` does exist,
                // `findEdgeAroundVertex` returning `none` means that `v` does
                // not have an adjacent boundary edge.
                //
                // A classical HEM can't deal with these kind of so called
                // non-manifold meshes. Marching cube won't create non-manifold
                // meshes, so for now we just panic. Having this restriction to
                // manifold meshes is probably not too bad.
                auto eEndH = findEdgeAroundVertex(vH, [this](auto edgeH)
                {
                    return static_cast<bool>(getE(edgeH).face);
                }).unwrap();

                auto eStartH = getE(eEndH).next;
                eIn.next = eStartH;
                getE(eEndH).next = eOutH;
            }
            else
            {
                // `eIn` and `eOut` are the only edges of the vertex.
                eIn.next = eOutH;
            }

        }
        // --> Case (B): only the ingoing edge is part of a face
        else if (eIn.face && !eOut.face)
        {
            // We know that `v` has at least two outgoing edges (since
            // there is a face adjacent to it).
            //
            // We have to find the edge which `next` handle pointed to the
            // outer edge of the one face we are adjacent to (called
            // `oldNext`. This is an inner edge of our face. This edge also
            // has to be a boundary edge or else we are dealing with a
            // non-manifold mesh again.
            //
            // Since we already know that `v.outgoing` does exist,
            // `findEdgeAroundVertex` returning `none` means that `v` does
            // not an edge which `next` handle points to `oldNext`. But this
            // has to be true in a non-broken HEM. So we will panic if this
            // condition is violated.
            auto eH = findEdgeAroundVertex(vH, [&, this](auto edgeH)
            {

                auto oldNext = eIn.twin;
                return !getE(edgeH).face && getE(edgeH).next == oldNext;
            }).unwrap();

            getE(eH).next = eOutH;
        }
        // --> Case (C): only the outgoing edge is part of a face
        else if (!eIn.face && eOut.face)
        {
            eIn.next = getE(eOut.twin).next;
        }
        // --> Case (D): both edges are already part of another face
        if (eIn.face && eOut.face)
        {
            // Nothing needs to be done!
        }
    }

    // Set `next` handle of inner edges. This is an easy step, but we can't
    // do it earlier, since the old `next` handles are required by the
    // previous "fix next handles" step.
    eInner1.next = eInner2H;
    eInner2.next = eInner3H;
    eInner3.next = eInner1H;

    // Set outgoing-handles if they are not set yet.
    if (!v1.outgoing)
    {
        v1.outgoing = eInner1H;
    }
    if (!v2.outgoing)
    {
        v2.outgoing = eInner2H;
    }
    if (!v3.outgoing)
    {
        v3.outgoing = eInner3H;
    }


    // Debug output
    cout << "+------ Summary face " << newFaceH << " ------+" << endl;
    auto startEdgeH = f.edge;
    auto eH = startEdgeH;
    int i = 0;
    do
    {
        auto& e = getE(eH);
        auto source = getE(e.twin).target;
        auto target = e.target;
        cout << "| " << source.idx() << " ==> " << eH
             << " ==> " << target.idx() << " [next: " << e.next
             << ", twin-next: " << getE(e.twin).next << "]"
             << endl;

        eH = e.next;
        i++;
        if (i >= 4)
        {
            cout << "DETECTED BUG" << endl;
            throw false;
        }
    } while(eH != startEdgeH);
    cout << "+-----------------------------+" << endl;
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
const typename HalfEdgeMesh<BaseVecT>::Edge&
    HalfEdgeMesh<BaseVecT>::getE(EdgeHandle handle) const
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
const typename HalfEdgeMesh<BaseVecT>::Face&
    HalfEdgeMesh<BaseVecT>::getF(FaceHandle handle) const
{
    return m_faces[handle.idx()];
}

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Vertex&
    HalfEdgeMesh<BaseVecT>::getV(VertexHandle handle)
{
    return m_vertices[handle.idx()];
}

template<typename BaseVecT>
const typename HalfEdgeMesh<BaseVecT>::Vertex&
    HalfEdgeMesh<BaseVecT>::getV(VertexHandle handle) const
{
    return m_vertices[handle.idx()];
}

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::edgeBetween(VertexHandle fromH, VertexHandle toH)
{
    return findEdgeAroundVertex(fromH, [&, this](auto edgeH)
    {
        return getE(getE(edgeH).twin).target == toH;
    });
    auto& from = getV(fromH);
}

template <typename BaseVecT>
typename BaseMesh<BaseVecT>::EdgeHandle
    HalfEdgeMesh<BaseVecT>::findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH)
{
    cout << "# findOrCreateEdgeBetween: " << fromH << " --> " << toH << endl;
    auto foundEdge = edgeBetween(fromH, toH);
    if (foundEdge)
    {
        cout << ">> found: " << foundEdge << endl;
        return foundEdge.unwrap();
    }
    else
    {
        cout << ">> adding pair..." << endl;
        return addEdgePair(fromH, toH).first;
    }
}

template <typename BaseVecT>
template <typename Pred>
typename BaseMesh<BaseVecT>::OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::findEdgeAroundVertex(VertexHandle vH, Pred pred) const
{
    // This function simply follows `next` and `twin` handles to visit all
    // edges around a vertex.
    cout << ">> Trying to find an edge around " << vH << " ..." << endl;

    auto& v = getV(vH);
    if (!v.outgoing)
    {
        cout << ">> ... " << vH << " has no outgoing edge, returning none." << endl;
        return OptionalEdgeHandle();
    }

    const auto startEdgeH = getE(v.outgoing.unwrap()).twin;
    auto loopEdgeH = startEdgeH;

    cout << ">> ... start search at outgoing " << v.outgoing.unwrap() << endl;
    cout << ">> ... startEdgeH: " << startEdgeH << endl;

    while (!pred(loopEdgeH))
    {
        cout << ">> ... >> LOOP: loop-" << loopEdgeH << " @ "
             << getE(loopEdgeH).face
             << " with next: " << getE(loopEdgeH).next << endl;

        loopEdgeH = getE(getE(loopEdgeH).next).twin;
        if (loopEdgeH == startEdgeH)
        {
            cout << ">> ... we visited all edges once without success, returning none." << endl;
            return OptionalEdgeHandle();
        }
    }
    cout << ">> ... found " << loopEdgeH << "." << endl;
    return loopEdgeH;
}

template <typename BaseVecT>
pair<typename BaseMesh<BaseVecT>::EdgeHandle, typename BaseMesh<BaseVecT>::EdgeHandle>
    HalfEdgeMesh<BaseVecT>::addEdgePair(VertexHandle v1H, VertexHandle v2H)
{
    // This method adds two new half edges, called "a" and "b".
    //
    //  +----+  --------(a)-------->  +----+
    //  | v1 |                        | v2 |
    //  +----+  <-------(b)---------  +----+

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

    // Add edges to our edge list.
    m_edges.push_back(a);
    m_edges.push_back(b);

    return std::make_pair(aH, bH);
}

template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::countVertices()
{
    return m_vertices.size();
}

template <typename BaseVecT>
Point<BaseVecT> HalfEdgeMesh<BaseVecT>::getPoint(size_t vertexIdx)
{
    return m_vertices[vertexIdx].pos;
}


} // namespace lvr

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
    using std::pair;
    using std::make_pair;

    // Handles for the inner edges of the face
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


    // Create face
    FaceHandle newFaceH = m_faces.size();

    // Calc normal
    // TODO: move to seperate method
    // TODO: fix copy&paste bug
    auto v1 = getV(v1H).pos;
    auto v2 = getV(v1H).pos;
    auto v3 = getV(v1H).pos;
    auto normal = (v1 - v2).cross(v1 - v3);

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

        // --> Case (D): both edges are already part of another face
        if (eIn.face && eOut.face)
        {
            // Nothing needs to be done: ...
        }
        // --> Case (A): neither edge is part of a face (both edges are new)
        else if (!eIn.face && !eOut.face)
        {
            // First we need to find two boundary edges adjacent to `v`. One
            // ending at `v`, called eEnd, and one starting at `v`, called
            // eStart. Note that `eEnd.next == eStart`.
            //
            // We search by starting at `v.outgoing` and iterating by following
            // the `.twin.next` handles. However, we need to handle the special
            // case of `v` not having an outgoing edge.
            if (v.outgoing)
            {
                const auto startEdgeH = getE(v.outgoing.unwrap()).twin;
                auto loopEdgeH = startEdgeH;
                // auto loopEdgeNextH = getE(loopEdgeH).next;

                cout << ">> start search at outgoing " << v.outgoing.unwrap() << endl;
                cout << ">> startEdgeH: " << startEdgeH << endl;

                while (getE(loopEdgeH).face)
                {
                    cout << ">>>> LOOP: loop-" << loopEdgeH << " @ "
                         << getE(loopEdgeH).face.unwrap().idx()
                         << " with next: " << getE(loopEdgeH).next << endl;
                    // auto target = getE(boundaryEdgeH).target;
                    // auto source = getE(getE(boundaryEdgeH).twin).target;
                    // cout << ">> visiting " << boundaryEdgeH
                    //      << " (" << source.idx()  << " <-> " << vH << ")" << endl;
                    loopEdgeH = getE(getE(loopEdgeH).next).twin;
                    // loopEdgeH = getE(loopEdgeNextH).twin;
                    // loopEdgeNextH = getE(loopEdgeH).next;
                    if (loopEdgeH == startEdgeH)
                    {
                        // In this case a non-manifold vertex would be created. We
                        // could potentially easily fix it by duplicating the vertex
                        // and treating both vertices as distinct ones. But for now,
                        // we simply assert that this never happens.
                        assert(false);
                        throw false;
                    }
                }
                cout << ">> found " << loopEdgeH << endl;

                auto eEndH = loopEdgeH;
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
        // --> Case (B):
        else if (eIn.face && !eOut.face)
        {
            // We know that `v` has at least two outgoing edges (since there
            // is a face adjacent to it).
            //
            // We have to find the edge which `next` handled pointed to the
            // one face we are adjacent to.
            auto expected = eIn.twin;

            const auto startEdgeH = getE(v.outgoing.unwrap()).twin;
            auto loopEdgeH = startEdgeH;

            cout << ">> start search at outgoing " << v.outgoing.unwrap() << endl;
            cout << ">> startEdgeH: " << startEdgeH << endl;

            while (!(!getE(loopEdgeH).face && getE(loopEdgeH).next == expected))
            {
                cout << ">>>> LOOP: loop-" << loopEdgeH << " @ "
                     << getE(loopEdgeH).face.unwrap().idx()
                     << " with next: " << getE(loopEdgeH).next << endl;
                loopEdgeH = getE(getE(loopEdgeH).next).twin;

                if (loopEdgeH == startEdgeH)
                {
                    // In this case a non-manifold vertex would be created. We
                    // could potentially easily fix it by duplicating the vertex
                    // and treating both vertices as distinct ones. But for now,
                    // we simply assert that this never happens.
                    assert(false);
                    throw false;
                }
            }
            cout << ">> found " << loopEdgeH << endl;

            getE(loopEdgeH).next = eOutH;
        }
        else if (!eIn.face && eOut.face)
        {
            eIn.next = getE(eOut.twin).next;
        }
    }


    // Set `next` handle of inner edges
    eInner1.next = eInner2H;
    eInner2.next = eInner3H;
    eInner3.next = eInner1H;


    // // Fix `next` handles that might be invalid after adding all edges
    // // independent from another.
    // if (getE(e3H).next != e1H)
    // {
    //     // We know that after we added the edges, `v1` has at least one
    //     // outgoing edge.
    //     auto boundaryEdgeH = boundaryEdgeOf(v1H).unwrap();

    //     auto prevNextH = getE(e3H).next;
    //     cout << "FIXING..." << endl
    //          << ">>> setting " << e3H << ".next = " << e1H << endl
    //          << ">>> setting " << getE(e1H).twin << ".next = " << prevNextH << endl
    //          << ">>> setting " << boundaryEdgeH << ".next = " << getE(e3H).twin << endl;
    //     getE(e3H).next = e1H;
    //     getE(getE(e1H).twin).next = prevNextH;
    //     getE(boundaryEdgeH).next = getE(e3H).twin;
    // }

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

    // if (!getV(v1H).outgoing)
    // {
    //     getV(v1H).outgoing = aH;
    // }
    // if (!getV(v2H).outgoing)
    // {
    //     getV(v2H).outgoing = bH;
    // }

    // Add edges to vector.
    m_edges.push_back(a);
    m_edges.push_back(b);

    // auto fixNextHandles = [&](VertexHandle vH, EdgeHandle ingoingH, EdgeHandle outgoingH)
    // {
    //     auto& v = getV(vH);
    //     if (v.outgoing)
    //     {
    //         // We already checked that `v` has an outgoing edge, so we can
    //         // unwrap.
    //         auto boundaryEdgeH = boundaryEdgeOf(vH).unwrap();

    //         // Visualization:
    //         //
    //         //
    //         //             ^  ...  /
    //         //  (prevNext)  \     /  (boundaryEdge)
    //         //               \   /
    //         //                \ v
    //         //                [v]
    //         //                ^ |
    //         //                | |
    //         //           (in) | | (out)
    //         //                | v
    //         //
    //         //
    //         auto prevNextH = getE(boundaryEdgeH).next;
    //         cout << ">>>> setting " << boundaryEdgeH << ".next = " << outgoingH << endl;
    //         cout << ">>>> setting " << ingoingH << ".next = " << prevNextH << endl;
    //         getE(boundaryEdgeH).next = outgoingH;
    //         getE(ingoingH).next = prevNextH;
    //     }
    //     else
    //     {
    //         // This is the easy case: the vertex was not connected before, so
    //         // we don't need to change any handles of other edges.
    //         //
    //         //         [v]
    //         //         ^ |
    //         //         | |
    //         //    (in) | | (out)
    //         //         | v
    //         //
    //         v.outgoing = outgoingH;
    //         getE(ingoingH).next = outgoingH;
    //     }
    // };

    // fixNextHandles(v1H, bH, aH);
    // fixNextHandles(v2H, aH, bH);

    return std::make_pair(aH, bH);
}


// template <typename BaseVecT>
// typename BaseMesh<BaseVecT>::OptionalEdgeHandle
//     HalfEdgeMesh<BaseVecT>::boundaryEdgeOf(VertexHandle vH)
// {
//     cout << "trying to find boundary edge of " << vH << endl;
//     auto& v = getV(vH);
//     if (v.outgoing)
//     {
//         cout << ">> start search at outgoing " << v.outgoing.unwrap() << endl;
//         const auto startEdgeH = getE(v.outgoing.unwrap()).twin;
//         cout << ">> startEdgeH: " << startEdgeH << endl;
//         auto boundaryEdgeH = startEdgeH;

//         auto continueSearch = [this](auto iEdgeH)
//         {
//             auto& e = getE(iEdgeH);
//             auto nextTwinH = getE(e.next).twin;
//             return e.face || (nextTwinH != iEdgeH && !getE(nextTwinH).face);
//         };

//         while (continueSearch(boundaryEdgeH))
//         {
//             cout << ">>>> LOOP: i-" << boundaryEdgeH << " @ "
//                  << getE(boundaryEdgeH).face.unwrap().idx()
//                  << " with next: " << getE(boundaryEdgeH).next << endl;
//             // auto target = getE(boundaryEdgeH).target;
//             // auto source = getE(getE(boundaryEdgeH).twin).target;
//             // cout << ">> visiting " << boundaryEdgeH
//             //      << " (" << source.idx()  << " <-> " << vH << ")" << endl;
//             boundaryEdgeH = getE(getE(boundaryEdgeH).next).twin;
//             if (boundaryEdgeH == startEdgeH)
//             {
//                 // In this case a non-manifold vertex would be created. We
//                 // could potentially easily fix it by duplicating the vertex
//                 // and treating both vertices as distinct ones. But for now,
//                 // we simply assert that this never happens.
//                 assert(false);
//                 throw false;
//             }
//         }
//         cout << ">> found " << boundaryEdgeH << endl;
//         return boundaryEdgeH;
//     }
//     cout << ">> none found" << endl;
//     return OptionalEdgeHandle();
// }

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

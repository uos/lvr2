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

#include <algorithm>
#include <array>
#include <utility>
#include <iostream>

#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/util/Panic.hpp>
#include <lvr2/util/Debug.hpp>


namespace lvr2
{


// ========================================================================
// = Interface methods
// ========================================================================

template <typename BaseVecT>
VertexHandle HalfEdgeMesh<BaseVecT>::addVertex(Point<BaseVecT> pos)
{
    Vertex v;
    v.pos = pos;
    return m_vertices.push(v);
}

template <typename BaseVecT>
FaceHandle HalfEdgeMesh<BaseVecT>::addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H)
{
    if (!BaseMesh<BaseVecT>::isFaceInsertionValid(v1H, v2H, v3H))
    {
        panic("Attempting to add a face which cannot be added!");
    }

    using std::make_tuple;

    DOINDEBUG(dout() << "##################################################" << endl);
    DOINDEBUG(dout() << "##### addFace(): " << v1H << " -> " << v2H << " -> " << v3H << endl);


    // =======================================================================
    // = Create broken edges
    // =======================================================================
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

    auto& v1 = getV(v1H);
    auto& v2 = getV(v2H);
    auto& v3 = getV(v3H);


    // =======================================================================
    // = Create face
    // =======================================================================
    FaceHandle newFaceH = m_faces.nextHandle();
    Face f(eInner1H);
    m_faces.push(f);

    // Set face handle of edges
    eInner1.face = newFaceH;
    eInner2.face = newFaceH;
    eInner3.face = newFaceH;


    // =======================================================================
    // = Fix next handles and set outgoing handles if not set yet
    // =======================================================================
    // Fixing the `next` handles is the most difficult part of this method. In
    // order to tackle it we deal with each corner of this face on its own.
    // For each corner we look at the corner-vertex and the in-going and
    // out-going edge (both edges are on the outside of this face!).
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

        auto& eIn  = getE(eInH);
        auto& v    = getV(vH);
        auto& eOut = getE(eOutH);

        // For each corner, we have for different cases, depending on whether
        // or not the in/outgoing edges are already part of a face.
        //
        // --> Case (A): neither edge is part of a face (both edges are new)
        if (!eIn.face && !eOut.face)
        {
            DOINDEBUG(dout() << "Case (A) for " << vH << endl);

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
                    return !getE(edgeH).face;
                }).unwrap();

                auto eStartH = getE(eEndH).next;
                eIn.next = eStartH;
                getE(eEndH).next = eOutH;

                DOINDEBUG(dout() << "(A) ... setting " << eInH << ".next = " << eStartH << endl);
                DOINDEBUG(dout() << "(A) ... setting " << eEndH << ".next = " << eOutH << endl);
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
            DOINDEBUG(dout() << "Case (B) for " << vH << endl);

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

            DOINDEBUG(dout() << "(B) ... setting " << eH << ".next = " << eOutH << endl);
            getE(eH).next = eOutH;
        }
        // --> Case (C): only the outgoing edge is part of a face
        else if (!eIn.face && eOut.face)
        {
            DOINDEBUG(dout() << "Case (C) for " << vH << endl);
            DOINDEBUG(
                dout() << "(C) ... setting " << eInH << ".next = "
                    << getE(eOut.twin).next << endl
            );
            eIn.next = getE(eOut.twin).next;
        }
        // --> Case (D): both edges are already part of another face
        if (eIn.face && eOut.face)
        {
            DOINDEBUG(
                dout() << "Case (D) for " << vH
                   << "(eIn = " << eInH << ", eOut = " << eOutH << ")" << endl
            );

            // Here, two fan blades around `v` will be connected. Both blades
            // need to be in the right order for this to work. The order is
            // given by the `next` handles of the edges with the target `v`.
            // By following those handles (and `twin` handles), we can
            // circulate around `v`.
            //
            // When circulating, both fan blades need to be next to each other.
            // If that's not the case, we need to fix a few `next` handles. Not
            // being in the right order is caused by case (A), but it can't be
            // avoided there. Only when connecting the blades here, we can know
            // how to create the `next` circle.
            if (getE(eOut.twin).next != eIn.twin)
            {
                // Here we need to conceptually delete one fan blade from the
                // `next` circle around `v` and re-insert it into the right
                // position. We choose to "move" the fan blade starting with
                // `eIn`.
                //
                // The most difficult part is to find the edge that points to
                // `eIn.twin`. We could reach it easily if we would store
                // `previous` handles; finding it without those handles is
                // possible by circulating around the vertex.
                auto inactiveBladeEndH = findEdgeAroundVertex(vH, [&, this](auto edgeH)
                {
                    return getE(edgeH).next == eIn.twin;
                }).unwrap();

                // Instead of pointing to `eIn.twin`, it needs to "skip" the
                // `eIn` blade and point to the blade afterwards. So we need to
                // find the end of the `eIn` blade. Its next handle is the one
                // `inactiveBladeEnd` needs to point to.
                auto eInBladeEndH = findEdgeAroundVertex(
                    eInH,
                    [&, this](auto edgeH)
                    {
                        return !getE(edgeH).face;
                    }
                ).unwrap();

                // We can finally set the next pointer to skip the `eIn`
                // blade. After this line, circulating around `v` will work
                // but skip the whole `eIn` blade.
                getE(inactiveBladeEndH).next = getE(eInBladeEndH).next;

                // Now we need to re-insert it again. Fortunately, this is
                // easier. After this line, the circle is broken, but it will
                // be repaired by repairing the `next` handles within the face
                // later.
                getE(eInBladeEndH).next = getE(eOut.twin).next;

                DOINDEBUG(
                    dout() << "(D) ... setting " << inactiveBladeEndH << ".next = "
                    << getE(eInBladeEndH).next << endl
                );
                DOINDEBUG(
                    dout() << "(D) ... setting " << eInBladeEndH << ".next = "
                    << getE(eOut.twin).next << endl
                );
            }
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


    // =======================================================================
    // = Debug output
    // =======================================================================
    DOINDEBUG(dout() << "+------ Summary face " << newFaceH << " ------+" << endl);
    auto startEdgeH = f.edge;
    auto eH = startEdgeH;
    int i = 0;
    do
    {
        auto& e = getE(eH);
        auto source = getE(e.twin).target;
        auto target = e.target;
        DOINDEBUG(
            dout() << "| " << source.idx() << " ==> " << eH
               << " ==> " << target.idx() << " [next: " << e.next
               << ", twin-next: " << getE(e.twin).next << "]"
               << endl
        );

        eH = e.next;
        i++;
        if (i >= 4)
        {
            panic("bug in HEM: face with invalid internal next handles added");
        }
    } while(eH != startEdgeH);
    DOINDEBUG(dout() << "+-----------------------------+" << endl);

    return newFaceH;
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::removeFace(FaceHandle handle)
{
    // Marker vertices, to save the vertices and edges which will be deleted
    vector<HalfEdgeHandle> edgesToRemove;
    vector<VertexHandle> verticesToRemove;

    // Marker vertex to fix next pointer of edges. The next pointer of the
    // pair.first has to be set to pair.second (i.e. getE(pair.first).next =
    // pair.second)
    vector<pair<HalfEdgeHandle, HalfEdgeHandle>> fixNext;

    // Walk around inner edges of face and check connected edges and vertices.
    // If they are connected to other faces or edges, fix their links otherwise
    // mark them to be deleted.
    auto innerEdges = getInnerEdges(handle);
    for (auto edgeH: innerEdges)
    {
        // Get the current edge pair (i.e. the current inner edge and its twin)
        auto& edge = getE(edgeH);
        auto twin = getE(edge.twin);

        // Check if target vertex (v1) of current inner edge (b) can be
        // deleted. This is true, if b.twin == a
        //  +----+  --------(a)-------->  +----+
        //  | v1 |                        | v2 |
        //  +----+  <-------(b)---------  +----+
        //   ∧  |
        //   |  |
        //  (c)(d)
        //   |  |
        //   |  ∨
        //  +----+
        //  | v3 |
        //  +----+
        auto nextOutgoingEdgeH = getE(getE(edge.next).twin).next;
        if (nextOutgoingEdgeH == edge.twin)
        {
            verticesToRemove.push_back(edge.target);
        }
        else
        {
            // Target vertex cannot be deleted, because other edges are
            // connected to it. Fix the outgoing point of the vertex and set it
            // to the next outgoing edge
            auto& target = getV(edge.target);
            target.outgoing = nextOutgoingEdgeH;
        }

        // Check if current inner edge and its twin can be deleted
        if (twin.face)
        {
            // If our twin has a face, the current edge pair is still needed!
            // We only need to fix the face pointer of the current inner edge
            edge.face = OptionalFaceHandle();
        }
        else
        {
            // Mark the edge pair as deleted
            edgesToRemove.push_back(edgeH);
            edgesToRemove.push_back(edge.twin);

            // Find edges around target and source vertex, which point to the
            // current edge pair
            auto frontEdgeToFixH = findEdgeAroundVertex(edge.target, [&, this](auto eH)
            {
                return getE(eH).next == edge.twin;
            });
            auto backEdgeToFixH = findEdgeAroundVertex(twin.target, [&, this](auto eH)
            {
                return getE(eH).next == edgeH;
            });

            // If found, fix the next pointer of the edges by setting it to the
            // next outgoing edge of the vertex
            if (frontEdgeToFixH)
            {
                // The next outgoing vertex might not be safe, if it belongs to
                // the face, which we want to delete. If this is the case, take
                // the second next outgoing vertex
                auto next = getE(edge.next);
                auto nextTwin = getE(next.twin);
                if (!nextTwin.face)
                {
                    fixNext.push_back(make_pair(frontEdgeToFixH.unwrap(), nextTwin.next));
                }
                else
                {
                    fixNext.push_back(make_pair(frontEdgeToFixH.unwrap(), edge.next));
                }
            }
            if (backEdgeToFixH)
            {
                fixNext.push_back(make_pair(backEdgeToFixH.unwrap(), twin.next));
            }
        }
    }

    // Fix next pointer
    for (auto pair: fixNext)
    {
        auto& edgeToFix = getE(pair.first);
        edgeToFix.next = pair.second;
    }

    // Actually remove the face and connected edges and vertices
    for (auto vertexH: verticesToRemove)
    {
        m_vertices.erase(vertexH);
    }
    for (auto edgeH: edgesToRemove)
    {
        m_edges.erase(edgeH);
    }
    m_faces.erase(handle);
}

template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::numVertices() const
{
    return m_vertices.numUsed();
}

template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::numFaces() const
{
    return m_faces.numUsed();
}

template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::numEdges() const
{
    return m_edges.numUsed() / 2;
}

template <typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::containsVertex(VertexHandle vH) const
{
    return static_cast<bool>(m_vertices.get(vH));
}

template <typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::containsFace(FaceHandle fH) const
{
    return static_cast<bool>(m_faces.get(fH));
}

template <typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::containsEdge(EdgeHandle eH) const
{
    return static_cast<bool>(m_edges.get(HalfEdgeHandle::oneHalfOf(eH)));
}

template <typename BaseVecT>
Index HalfEdgeMesh<BaseVecT>::nextVertexIndex() const
{
    return m_vertices.nextHandle().idx();
}

template <typename BaseVecT>
Index HalfEdgeMesh<BaseVecT>::nextFaceIndex() const
{
    return m_faces.nextHandle().idx();
}

template <typename BaseVecT>
Index HalfEdgeMesh<BaseVecT>::nextEdgeIndex() const
{
    return m_edges.nextHandle().idx();
}


template <typename BaseVecT>
Point<BaseVecT> HalfEdgeMesh<BaseVecT>::getVertexPosition(VertexHandle handle) const
{
    return getV(handle).pos;
}

template <typename BaseVecT>
Point<BaseVecT>& HalfEdgeMesh<BaseVecT>::getVertexPosition(VertexHandle handle)
{
    return getV(handle).pos;
}

template <typename BaseVecT>
array<VertexHandle, 3> HalfEdgeMesh<BaseVecT>::getVerticesOfFace(FaceHandle handle) const
{
    auto face = getF(handle);

    auto e1 = getE(face.edge);
    auto e2 = getE(e1.next);
    auto e3 = getE(e2.next);

    return {e1.target, e2.target, e3.target};
}

template <typename BaseVecT>
array<EdgeHandle, 3> HalfEdgeMesh<BaseVecT>::getEdgesOfFace(FaceHandle handle) const
{
    auto innerEdges = getInnerEdges(handle);
    return {
        halfToFullEdgeHandle(innerEdges[0]),
        halfToFullEdgeHandle(innerEdges[1]),
        halfToFullEdgeHandle(innerEdges[2])
    };
}

template <typename BaseVecT>
array<HalfEdgeHandle, 3> HalfEdgeMesh<BaseVecT>::getInnerEdges(FaceHandle handle) const
{
    auto face = getF(handle);

    // Get inner edges in counter clockwise order
    auto e1 = getE(face.edge);
    auto e2 = getE(e1.next);

    return {face.edge, e1.next, e2.next};
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::getNeighboursOfFace(
    FaceHandle handle,
    vector<FaceHandle>& facesOut
) const
{
    auto face = getF(handle);

    // Get inner edges in counter clockwise order
    auto e1 = getE(face.edge);
    auto e2 = getE(e1.next);
    auto e3 = getE(e2.next);

    // Get twins of inner edges
    auto e1t = getE(e1.twin);
    auto e2t = getE(e2.twin);
    auto e3t = getE(e3.twin);

    // Get neighbour faces
    if (e1t.face)
    {
        facesOut.push_back(e1t.face.unwrap());
    }
    if (e2t.face)
    {
        facesOut.push_back(e2t.face.unwrap());
    }
    if (e3t.face)
    {
        facesOut.push_back(e3t.face.unwrap());
    }
}

template <typename BaseVecT>
array<VertexHandle, 2> HalfEdgeMesh<BaseVecT>::getVerticesOfEdge(EdgeHandle edgeH) const
{
    auto oneEdgeH = HalfEdgeHandle::oneHalfOf(edgeH);
    auto oneEdge = getE(oneEdgeH);
    return { oneEdge.target, getE(oneEdge.twin).target };
}

template <typename BaseVecT>
array<OptionalFaceHandle, 2> HalfEdgeMesh<BaseVecT>::getFacesOfEdge(EdgeHandle edgeH) const
{
    auto oneEdgeH = HalfEdgeHandle::oneHalfOf(edgeH);
    auto oneEdge = getE(oneEdgeH);
    return { oneEdge.face, getE(oneEdge.twin).face };
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::getFacesOfVertex(
    VertexHandle handle,
    vector<FaceHandle>& facesOut
) const
{
    circulateAroundVertex(handle, [&facesOut, this](auto eH)
    {
        auto edge = getE(eH);
        if (edge.face)
        {
            facesOut.push_back(edge.face.unwrap());
        }
        return true;
    });
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::getEdgesOfVertex(
    VertexHandle handle,
    vector<EdgeHandle>& edgesOut
) const
{
    circulateAroundVertex(handle, [&edgesOut, this](auto eH)
    {
        edgesOut.push_back(halfToFullEdgeHandle(eH));
        return true;
    });
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::getNeighboursOfVertex(
    VertexHandle handle,
    vector<VertexHandle>& verticesOut
) const
{
    circulateAroundVertex(handle, [&verticesOut, this](auto eH)
    {
        verticesOut.push_back(getE(getE(eH).twin).target);
        return true;
    });
}

template <typename BaseVecT>
EdgeCollapseResult HalfEdgeMesh<BaseVecT>::collapseEdge(EdgeHandle edgeH)
{
    if (!BaseMesh<BaseVecT>::isCollapsable(edgeH))
    {
        panic("call to collapseEdge() with non-collapsable edge!");
    }

    // The general (and most common) case looks like this:
    //
    //             [C]                | Vertices:
    //             / ^                | [A]: vertexToKeep
    //            /   \               | [B]: vertexToRemove
    //         c /     \ b            | [C]: vertexAbove
    //          /       \             | [D]: vertexBelow
    //         V    a    \            |
    //           ------>              | Edges:
    //      [A]            [B]        | a: startEdge
    //           <------              | d: startEdgeTwin
    //         \    d    ^            |
    //          \       /             |
    //         e \     / f            |
    //            \   /               |
    //             V /                |
    //             [D]                |
    //
    // The variable naming in this method imagines that the given edge points
    // to the right and there might be one face above and one face below. The
    // start edges and the inner edges of those faces will be removed and for
    // each face the remaining two edges become twins.

    auto startEdgeH = HalfEdgeHandle::oneHalfOf(edgeH);
    auto& startEdge = getE(startEdgeH);
    auto startEdgeTwinH = startEdge.twin;
    auto& startEdgeTwin = getE(startEdgeTwinH);

    // The two vertices that are merged
    auto vertexToRemoveH = startEdge.target;
    auto vertexToKeepH = startEdgeTwin.target;

    // The two faces next to the given edge
    OptionalFaceHandle faceAboveH = startEdge.face;
    OptionalFaceHandle faceBelowH = startEdgeTwin.face;

    // The result that contains information about the removed faces and edges
    // and the new vertex and edges
    EdgeCollapseResult result(vertexToKeepH);

    // Fix targets for ingoing edges of the vertex that will be deleted. This
    // has to be done before changing the twin edges.
    circulateAroundVertex(vertexToRemoveH, [&, this](auto ingoingEdgeH)
    {
        getE(ingoingEdgeH).target = vertexToKeepH;
        return true;
    });

    // Save edges to delete for later
    optional<array<HalfEdgeHandle, 2>> edgesToDeleteAbove;
    optional<array<HalfEdgeHandle, 2>> edgesToDeleteBelow;

    // If there is a face above, collapse it.
    if (faceAboveH)
    {
        DOINDEBUG(dout() << "... has face above" << endl);

        // The situations looks like this now:
        //
        //             [C]                | Vertices:
        //          ^  / ^  \             | [A]: vertexToKeep
        //         /  /   \  \            | [B]: vertexToRemove
        //      x /  /c   b\  \ y         | [C]: vertexAbove
        //       /  /       \  \          |
        //      /  V    a    \  v         | Edges:
        //           ------>              | a: startEdge
        //      [A]            [B]        | b: edgeToRemoveAR (AboveRight)
        //           <------              | c: edgeToRemoveAL (AboveLeft)
        //                                | x: edgeToKeepAL (AboveLeft)
        //                                | y: edgeToKeepAR (AboveRight)
        //

        // Give names to important edges and vertices
        auto edgeToRemoveARH = startEdge.next;
        auto edgeToKeepARH = getE(edgeToRemoveARH).twin;
        auto edgeToRemoveALH = getE(edgeToRemoveARH).next;
        auto edgeToKeepALH = getE(edgeToRemoveALH).twin;

        auto vertexAboveH = getE(edgeToRemoveARH).target;

        // Fix twin edges ("fuse" edges)
        getE(edgeToKeepALH).twin = edgeToKeepARH;
        getE(edgeToKeepARH).twin = edgeToKeepALH;

        // Fix outgoing edges of vertices because they might be deleted
        getV(vertexToKeepH).outgoing = edgeToKeepALH;
        getV(vertexAboveH).outgoing = edgeToKeepARH;

        // Write handles to result
        result.neighbors[0] = EdgeCollapseRemovedFace(
            faceAboveH.unwrap(),
            {
                halfToFullEdgeHandle(edgeToRemoveARH),
                halfToFullEdgeHandle(edgeToRemoveALH)
            },
            halfToFullEdgeHandle(edgeToKeepALH)
        );
        
        std::array<lvr2::HalfEdgeHandle, 2> arr = {
            edgeToRemoveARH,
            edgeToRemoveALH
        }; 
        // We need to defer the actually removal...
        edgesToDeleteAbove = arr;
    }
    else
    {
        DOINDEBUG(dout() << "... doesn't have a face above" << endl);

        // The situation looks like this now:
        //
        //    \                    ^         | Vertices:
        //     \ c              b /          | [A]: vertexToKeep
        //      \       a        /           | [B]: vertexToRemove
        //       v   ------>    /            |
        //      [A]            [B]           | Edges:
        //           <------                 | a: startEdge
        //              d                    | b: startEdge.next
        //                                   | c: startEdgePrecursor
        //
        //
        // How do we know that it doesn't actually look like in the if-branch
        // and that c and b doesn't actually share a vertex? This is asserted
        // by the `isCollapsable()` method!
        //
        // If there is no triangle above, the edge whose next is the start edge
        // needs the correct new next edge
        auto startEdgePrecursorH = findEdgeAroundVertex(startEdgeTwinH, [&, this](auto eH)
        {
            return getE(eH).next == startEdgeH;
        });
        getE(startEdgePrecursorH.unwrap()).next = startEdge.next;

        // Fix outgoing edge of vertexToKeep because it might be deleted
        getV(vertexToKeepH).outgoing = startEdge.next;
    }

    if (faceBelowH)
    {
        DOINDEBUG(dout() << "... has face below" << endl);

        // The situation looks like this now:
        //
        //              a                 | Vertices:
        //           ------>              | [A]: vertexToKeep
        //      [A]            [B]        | [B]: vertexToRemove
        //           <------              | [D]: vertexBelow
        //      ^  \    d    ^  /         |
        //       \  \       /  /          | Edges:
        //      x \  \e   f/  / y         | d: startEdgeTwin
        //         \  \   /  /            | e: edgeToRemoveBL (BelowLeft)
        //          \  V /  v             | f: edgeToRemoveBR (BelowRight)
        //             [D]                | x: edgeToKeepBL (BelowLeft)
        //                                | y: edgeToKeepBR (BelowRight)
        //
        //
        //

        // Give names to important edges and vertices
        auto edgeToRemoveBLH = startEdgeTwin.next;
        auto edgeToKeepBLH = getE(edgeToRemoveBLH).twin;
        auto edgeToRemoveBRH = getE(edgeToRemoveBLH).next;
        auto edgeToKeepBRH = getE(edgeToRemoveBRH).twin;

        auto vertexBelowH = getE(edgeToRemoveBLH).target;

        // Fix twin edges ("fuse" edges)
        getE(edgeToKeepBLH).twin = edgeToKeepBRH;
        getE(edgeToKeepBRH).twin = edgeToKeepBLH;

        // Fix outgoing edge of vertexBelow because it might be deleted
        getV(vertexBelowH).outgoing = edgeToKeepBLH;

        // Write handles to result
        result.neighbors[1] = EdgeCollapseRemovedFace(
            faceBelowH.unwrap(),
            {
                halfToFullEdgeHandle(edgeToRemoveBRH),
                halfToFullEdgeHandle(edgeToRemoveBLH)
            },
            halfToFullEdgeHandle(edgeToKeepBLH)
        );
        
        std::array<lvr2::HalfEdgeHandle, 2> arr = {
            edgeToRemoveBRH,
            edgeToRemoveBLH
        };

        // We need to defer the actual removal...
        edgesToDeleteBelow = arr;
    }
    else
    {
        DOINDEBUG(dout() << "!(faceBelow || hasTriangleBelow)" << endl);

        // The situation looks like this now:
        //
        //              a                     | Vertices:
        //           ------>                  | [A]: vertexToKeep
        //      [A]            [B]            | [B]: vertexToRemove
        //       /   <------    ^             |
        //      /       d        \            | Edges:
        //     / e              f \           | d: startEdgeTwin
        //    v                    \          | e: startEdgeTwin.next
        //                                    | f: startEdgeTwinPrecursor
        //

        // If there is no triangle below, the edge whose next is the twin of the
        // start edge needs the correct new next edge
        auto startEdgeTwinPrecursorH = findEdgeAroundVertex(startEdgeH, [&, this](auto eH)
        {
            return getE(eH).next == startEdgeTwinH;
        });
        getE(startEdgeTwinPrecursorH.unwrap()).next = startEdgeTwin.next;
    }

    // Calculate and set new position of the vertex that is kept
    {
        auto position1 = getV(vertexToRemoveH).pos;
        auto position2 = getV(vertexToKeepH).pos;

        auto newPosition = position1 + (position2 - position1) / 2;
        getV(vertexToKeepH).pos = newPosition;
    }

    // Delete one vertex
    DOINDEBUG(dout() << "Remove vertex: " << vertexToRemoveH << endl);
    m_vertices.erase(vertexToRemoveH);

    // Delete edges and faces
    if (faceAboveH)
    {
        auto edgeToRemove0 = (*edgesToDeleteAbove)[0];
        auto edgeToRemove1 = (*edgesToDeleteAbove)[1];

        // Actually delete edges and the face above
        DOINDEBUG(
            dout() << "Remove face above with edges: " << faceAboveH << ", "
                << edgeToRemove0 << ", " << edgeToRemove1 << endl
        );

        m_edges.erase(edgeToRemove0);
        m_edges.erase(edgeToRemove1);
        m_faces.erase(faceAboveH.unwrap());
    }
    if (faceBelowH)
    {
        auto edgeToRemove0 = (*edgesToDeleteBelow)[0];
        auto edgeToRemove1 = (*edgesToDeleteBelow)[1];

        // Actually delete edges and the faces
        DOINDEBUG(
            dout() << "Remove face below with edges: " << faceBelowH << ", "
                << edgeToRemove0 << ", " << edgeToRemove1 << endl
        );

        m_edges.erase(edgeToRemove0);
        m_edges.erase(edgeToRemove1);
        m_faces.erase(faceBelowH.unwrap());
    }

    DOINDEBUG(dout() << "Remove start edges: " << startEdgeH << " and " << startEdge.twin << endl);
    m_edges.erase(startEdgeH);
    m_edges.erase(startEdgeTwinH);

    return result;
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::flipEdge(EdgeHandle edgeH)
{
    if (!BaseMesh<BaseVecT>::isFlippable(edgeH))
    {
        panic("flipEdge() called for non-flippable edge!");
    }

    // A fancy drawing of the current and expected situation:
    //
    //                                  |                                     |
    //            Before                |                After                |
    //            ------                |                -----                |
    //                                  |                                     |
    //                                  |                                     |
    //              [C]                 |                 [C]                 |
    //                                  |                                     |
    //          ^  /   ^  \             |             ^  /   ^  \             |
    //         /  /     \  \            |            /  / ^ | \  \            |
    //        /  /       \  \           |           /  /  | |  \  \           |
    //       /  /         \  \          |          /  /   | |   \  \          |
    //      /  /c         d\  \         |         /  /    | |    \  \         |
    //     /  /     (X)     \  \        |        /  /     | |     \  \        |
    //    /  /               \  \       |       /  /      | |      \  \       |
    //   /  /                 \  \      |      /  /       | |       \  \      |
    //  /  v         a         \  v     |     /  v        | |        \  v     |
    //       --------------->           |                 | |                 |
    //  [A]                     [B]     |     [A]   (X)   | |   (Y)   [B]     |
    //       <---------------           |                 | |                 |
    //  ^  \         b         ^  /     |     ^  \        | |        ^  /     |
    //   \  \                 /  /      |      \  \       | |       /  /      |
    //    \  \               /  /       |       \  \      | |      /  /       |
    //     \  \     (Y)     /  /        |        \  \     | |     /  /        |
    //      \  \e         f/  /         |         \  \    | |    /  /         |
    //       \  \         /  /          |          \  \   | |   /  /          |
    //        \  \       /  /           |           \  \  | |  /  /           |
    //         \  \     /  /            |            \  \ | v /  /            |
    //          \  v   /  v             |             \  v   /  v             |
    //                                  |                                     |
    //              [D]                 |                 [D]                 |
    //
    //
    // ### A mapping from graphic names to variable names:
    //
    //  Edges                      Vertices               Faces
    //  -----                      --------               -----
    //  a: center                  [A]: vLeft             (X): faceAbove
    //  b: centerTwin              [B]: vRight            (Y): faceBelow
    //  c: aboveLeft               [C]: vAbove
    //  d: aboveRight              [D]: vBelow
    //  e: belowLeft
    //  f: belowRight
    //
    //
    // We just imagine that the random half-edge we get from `oneHalfOf()` is
    // the edge `a` in the drawing. We call it `center`.
    //
    // First, we just obtain all the handles
    auto centerH = HalfEdgeHandle::oneHalfOf(edgeH);
    auto& center = getE(centerH);
    auto centerTwinH = center.twin;
    auto& centerTwin = getE(centerTwinH);

    auto faceAboveH = center.face.unwrap();
    auto faceBelowH = centerTwin.face.unwrap();

    auto aboveRightH = center.next;
    auto& aboveRight = getE(aboveRightH);
    auto aboveLeftH = aboveRight.next;
    auto& aboveLeft = getE(aboveLeftH);
    auto belowLeftH = centerTwin.next;
    auto& belowLeft = getE(belowLeftH);
    auto belowRightH = belowLeft.next;
    auto& belowRight = getE(belowRightH);

    auto vLeftH = centerTwin.target;
    auto vRightH = center.target;
    auto vAboveH = aboveRight.target;
    auto vBelowH = belowLeft.target;

    // And now we just change all the handles. It's fairly easy, since we don't
    // have to deal with any special cases.
    //
    // First, fix outgoing handles, since those might be wrong now.
    getV(vLeftH).outgoing = belowLeftH;
    getV(vRightH).outgoing = aboveRightH;

    // Next, fix the `edge` pointer of the faces, since those might be broken
    // as well.
    getF(faceAboveH).edge = centerH;
    getF(faceBelowH).edge = centerTwinH;

    // Fix the all time favorite: the next handle. We basically change the next
    // handle of all inner edges. First all edges around the left face (former
    // "above" face), then the right (former "below") one.
    center.next = aboveLeftH;
    aboveLeft.next = belowLeftH;
    belowLeft.next = centerH;

    centerTwin.next = belowRightH;
    belowRight.next = aboveRightH;
    aboveRight.next = centerTwinH;

    // The target handle of both center edges...
    center.target = vAboveH;
    centerTwin.target = vBelowH;

    // And finally, two edges belong to a new face now.
    aboveRight.face = faceBelowH;  // now right
    belowLeft.face = faceAboveH;   // now left
}

template <typename BaseVecT>
EdgeHandle HalfEdgeMesh<BaseVecT>::halfToFullEdgeHandle(HalfEdgeHandle handle) const
{
    auto twin = getE(handle).twin;
    // return the handle with the smaller index of the given half edge and its twin
    return EdgeHandle(min(twin.idx(), handle.idx()));
}

// ========================================================================
// = Other public methods
// ========================================================================
template <typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::debugCheckMeshIntegrity() const
{
    using std::endl;

    cout << endl;
    cout << "===============================" << endl;
    cout << "===     Integrity check     ===" << endl;
    cout << "===============================" << endl;

    bool error = false;

    // First: let's visit all faces
    cout << endl;
    cout << "+--------------------+" << endl;
    cout << "| Checking all faces |" << endl;
    cout << "+--------------------+" << endl;
    for (auto fH : m_faces)
    {
        cout << "== Checking Face " << fH << "..." << endl;
        auto startEdgeH = getF(fH).edge;
        auto eH = startEdgeH;
        int edgeCount = 0;
        do
        {
            auto& e = getE(eH);
            auto source = getE(e.twin).target;
            auto target = e.target;
            cout << "   | " << eH << ": " << source << " ==> " << target
                 << " [next: " << e.next << ", twin: " << e.twin
                 << ", twin-face: " << getE(e.twin).face << "]"
                 << endl;

            if (getE(eH).face != fH)
            {
                cout << "!!!!! Face handle of " << eH << " is " << getE(eH).face
                     << " instead of " << fH << "!!!!!" << endl;
                error = true;
            }

            eH = e.next;
            edgeCount++;
            if (edgeCount >= 20)
            {
                cout << "   ... stopping iteration after 20 edges." << endl;
                break;
            }
        } while(eH != startEdgeH);

        if (edgeCount != 3)
        {
            cout << "!!!!! More than 3 edges reached from " << fH << endl;
            error = true;
        }
    }

    // Next, we try to reach all boundary edges
    cout << endl;
    cout << "+-------------------------------------+" << endl;
    cout << "| Trying to walk on boundary edges... |" << endl;
    cout << "+-------------------------------------+" << endl;

    DenseAttrMap<HalfEdgeHandle, bool> visited(m_edges.size(), false);
    for (auto startEdgeH : m_edges)
    {
        auto loopEdgeH = startEdgeH;

        if (visited[startEdgeH] || getE(startEdgeH).face)
        {
            continue;
        }
        visited[startEdgeH] = true;

        cout << "== Starting at " << startEdgeH << endl;

        do
        {
            loopEdgeH = getE(loopEdgeH).next;
            const auto twinH = getE(loopEdgeH).twin;
            visited[loopEdgeH] = true;
            cout << "   | -> " << loopEdgeH << " [twin: " << twinH << " | "
                 << getE(twinH).target << " --> " << getE(loopEdgeH).target
                 << "]" << endl;
        } while(loopEdgeH != startEdgeH);
    }

    // Next, we list all vertices that are not connected to anything yet
    cout << endl;
    cout << "+-------------------------------+" << endl;
    cout << "| List of unconnected vertices: |" << endl;
    cout << "+-------------------------------+" << endl;

    for (auto vH : m_vertices)
    {
        if (!getV(vH).outgoing)
        {
            cout << "== " << vH << endl;
        }
    }

    return error;
}


// ========================================================================
// = Private helper methods
// ========================================================================

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Edge&
    HalfEdgeMesh<BaseVecT>::getE(HalfEdgeHandle handle)
{
    return m_edges[handle];
}

template<typename BaseVecT>
const typename HalfEdgeMesh<BaseVecT>::Edge&
    HalfEdgeMesh<BaseVecT>::getE(HalfEdgeHandle handle) const
{
    return m_edges[handle];
}

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Face&
    HalfEdgeMesh<BaseVecT>::getF(FaceHandle handle)
{
    return m_faces[handle];
}

template<typename BaseVecT>
const typename HalfEdgeMesh<BaseVecT>::Face&
    HalfEdgeMesh<BaseVecT>::getF(FaceHandle handle) const
{
    return m_faces[handle];
}

template<typename BaseVecT>
typename HalfEdgeMesh<BaseVecT>::Vertex&
    HalfEdgeMesh<BaseVecT>::getV(VertexHandle handle)
{
    return m_vertices[handle];
}

template<typename BaseVecT>
const typename HalfEdgeMesh<BaseVecT>::Vertex&
    HalfEdgeMesh<BaseVecT>::getV(VertexHandle handle) const
{
    return m_vertices[handle];
}

template <typename BaseVecT>
OptionalHalfEdgeHandle
    HalfEdgeMesh<BaseVecT>::edgeBetween(VertexHandle fromH, VertexHandle toH)
{
    auto twinOut = findEdgeAroundVertex(fromH, [&, this](auto edgeH)
    {
        return getE(getE(edgeH).twin).target == toH;
    });
    if (twinOut)
    {
        return getE(twinOut.unwrap()).twin;
    }
    else
    {
        return OptionalHalfEdgeHandle();
    }
}

template <typename BaseVecT>
HalfEdgeHandle
    HalfEdgeMesh<BaseVecT>::findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH)
{
    DOINDEBUG(dout() << "# findOrCreateEdgeBetween: " << fromH << " --> " << toH << endl);
    auto foundEdge = edgeBetween(fromH, toH);
    if (foundEdge)
    {
        DOINDEBUG(dout() << ">> found: " << foundEdge << endl);
        return foundEdge.unwrap();
    }
    else
    {
        DOINDEBUG(dout() << ">> adding pair..." << endl);
        return addEdgePair(fromH, toH).first;
    }
}

template <typename BaseVecT>
template <typename Visitor>
void HalfEdgeMesh<BaseVecT>::circulateAroundVertex(VertexHandle vH, Visitor visitor) const
{
    auto outgoing = getV(vH).outgoing;
    if (outgoing)
    {
        circulateAroundVertex(getE(outgoing.unwrap()).twin, visitor);
    }
}

template <typename BaseVecT>
template <typename Visitor>
void HalfEdgeMesh<BaseVecT>::circulateAroundVertex(HalfEdgeHandle startEdgeH, Visitor visitor) const
{
    auto loopEdgeH = startEdgeH;

    DOINDEBUG(
        int iterCount = 0;
        vector<HalfEdgeHandle> visited;
    )

    while (true)
    {
        // Call the visitor and stop, if the visitor tells us to.
        if (!visitor(loopEdgeH))
        {
            break;
        }

        // Advance to next edge and stop if it is the start edge.
        loopEdgeH = getE(getE(loopEdgeH).next).twin;
        if (loopEdgeH == startEdgeH)
        {
            break;
        }

        DOINDEBUG(
            iterCount++;
            if (iterCount > 30)
            {
                // We don't want to loop forever here. This might only happen if
                // the HEM contains a bug. We want to break the loop at some point,
                // but without paying the price of managing the `visited` vector
                // in the common case (no bug). So we start manage the vector
                // after 30 iterations
                if (std::find(visited.begin(), visited.end(), loopEdgeH) != visited.end())
                {
                    panic("bug in HEM: detected cycle while looping around vertex");
                }
                visited.push_back(loopEdgeH);
            }
        )
    }
}

template <typename BaseVecT>
template <typename Pred>
OptionalHalfEdgeHandle
    HalfEdgeMesh<BaseVecT>::findEdgeAroundVertex(VertexHandle vH, Pred pred) const
{
    // This function simply follows `next` and `twin` handles to visit all
    // edges around a vertex.
    DOINDEBUG(dout() << ">> Trying to find an edge around " << vH << " ..." << endl);

    auto& v = getV(vH);
    if (!v.outgoing)
    {
        DOINDEBUG(dout() << ">> ... " << vH << " has no outgoing edge, returning none." << endl);
        return OptionalHalfEdgeHandle();
    }

    return findEdgeAroundVertex(getE(v.outgoing.unwrap()).twin, pred);
}

template <typename BaseVecT>
template <typename Pred>
OptionalHalfEdgeHandle HalfEdgeMesh<BaseVecT>::findEdgeAroundVertex(
    HalfEdgeHandle startEdgeH,
    Pred pred
) const
{
    // This function simply follows `next` and `twin` handles to visit all
    // edges around a vertex.
    DOINDEBUG(dout() << ">> Trying to find an edge starting from " << startEdgeH << " ..." << endl);

    OptionalHalfEdgeHandle out;
    circulateAroundVertex(startEdgeH, [&, this](auto ingoingEdgeH)
    {
        DOINDEBUG(
            dout() << ">> ... >> LOOP: loop-" << ingoingEdgeH << " @ "
               << getE(ingoingEdgeH).face
               << " with next: " << getE(ingoingEdgeH).next << endl
        );
        if (pred(ingoingEdgeH))
        {
            out = ingoingEdgeH;
            return false;
        }
        return true;
    });

    if (!out)
    {
        DOINDEBUG(
            dout() << ">> ... we visited all edges once without success, returning none."
                << endl
        );
    }
    else
    {
        DOINDEBUG(dout() << ">> ... found " << out.unwrap() << "." << endl);
    }

    return out;
}

template <typename BaseVecT>
pair<HalfEdgeHandle, HalfEdgeHandle> HalfEdgeMesh<BaseVecT>::addEdgePair(VertexHandle v1H, VertexHandle v2H)
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
    HalfEdgeHandle aH(m_edges.size());
    HalfEdgeHandle bH(m_edges.size() + 1);

    // Assign twins to each other
    a.twin = bH;
    b.twin = aH;

    // Assign half-edge targets
    a.target = v2H;
    b.target = v1H;

    // Add edges to our edge list.
    m_edges.push(a);
    m_edges.push(b);

    return std::make_pair(aH, bH);
}


// ========================================================================
// = Iterator stuff
// ========================================================================
template<typename HandleT, typename ElemT>
HemFevIterator<HandleT, ElemT>& HemFevIterator<HandleT, ElemT>::operator++()
{
    ++m_iterator;
    return *this;
}

template<typename HandleT, typename ElemT>
bool HemFevIterator<HandleT, ElemT>::operator==(const MeshHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HemFevIterator<HandleT, ElemT>*>(&other);
    return cast && m_iterator == cast->m_iterator;
}

template<typename HandleT, typename ElemT>
bool HemFevIterator<HandleT, ElemT>::operator!=(const MeshHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HemFevIterator<HandleT, ElemT>*>(&other);
    return !cast || m_iterator != cast->m_iterator;
}

template<typename HandleT, typename ElemT>
HandleT HemFevIterator<HandleT, ElemT>::operator*() const
{
    return *m_iterator;
}

template<typename BaseVecT>
HemEdgeIterator<BaseVecT>& HemEdgeIterator<BaseVecT>::operator++()
{
    ++m_iterator;

    // If not at the end, find the next half edge handle that equals the full edge handle of that edge
    // according to the halfToFullEdgeHandle method
    while (!m_iterator.isAtEnd() && (*m_iterator).idx() != m_mesh.halfToFullEdgeHandle(*m_iterator).idx())
    {
        ++m_iterator;
    }

    return *this;
}

template<typename BaseVecT>
bool HemEdgeIterator<BaseVecT>::operator==(const MeshHandleIterator<EdgeHandle>& other) const
{
    auto cast = dynamic_cast<const HemEdgeIterator<BaseVecT>*>(&other);
    return cast && m_iterator == cast->m_iterator;
}

template<typename BaseVecT>
bool HemEdgeIterator<BaseVecT>::operator!=(const MeshHandleIterator<EdgeHandle>& other) const
{
    auto cast = dynamic_cast<const HemEdgeIterator<BaseVecT>*>(&other);
    return !cast || m_iterator != cast->m_iterator;
}

template<typename BaseVecT>
EdgeHandle HemEdgeIterator<BaseVecT>::operator*() const
{
    return m_mesh.halfToFullEdgeHandle(*m_iterator);
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> HalfEdgeMesh<BaseVecT>::verticesBegin() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<HemFevIterator<VertexHandle, Vertex>>(this->m_vertices.begin())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> HalfEdgeMesh<BaseVecT>::verticesEnd() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<HemFevIterator<VertexHandle, Vertex>>(this->m_vertices.end())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> HalfEdgeMesh<BaseVecT>::facesBegin() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<HemFevIterator<FaceHandle, Face>>(this->m_faces.begin())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> HalfEdgeMesh<BaseVecT>::facesEnd() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<HemFevIterator<FaceHandle, Face>>(this->m_faces.end())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> HalfEdgeMesh<BaseVecT>::edgesBegin() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<HemEdgeIterator<BaseVecT>>(this->m_edges.begin(), *this)
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> HalfEdgeMesh<BaseVecT>::edgesEnd() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<HemEdgeIterator<BaseVecT>>(this->m_edges.end(), *this)
    );
}

} // namespace lvr2

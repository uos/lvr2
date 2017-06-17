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

#include <lvr2/util/VectorMap.hpp>
#include <lvr2/util/Panic.hpp>


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
    m_vertices.push_back(v);
    return VertexHandle(static_cast<Index>(m_vertices.size() - 1));
}

template <typename BaseVecT>
FaceHandle HalfEdgeMesh<BaseVecT>::addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H)
{
    using std::make_tuple;

    dout() << "##################################################" << endl;
    dout() << "##### addFace(): " << v1H << " -> " << v2H << " -> " << v3H << endl;


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
    // Calc normal
    auto normal = (v1.pos - v2.pos).cross(v1.pos - v3.pos);

    FaceHandle newFaceH = m_faces.size();
    Face f(eInner1H, Normal<BaseVecT>(normal));
    m_faces.push_back(f);

    // Set face handle of edges
    eInner1.face = newFaceH;
    eInner2.face = newFaceH;
    eInner3.face = newFaceH;


    // =======================================================================
    // = Fix next handles and set outgoing handles if net set yet
    // =======================================================================
    // Fixing the `next` handles is the most difficult part of this method. In
    // order to tackle it we deal with each corner of this face on its own.
    // For each corner we look at the corner-vertex and the in-going and
    //  out-going edge (both edges are on the outside of this face!).
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
            dout() << "Case (A) for " << vH << endl;

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

                dout() << "(A) ... setting " << eInH << ".next = " << eStartH << endl;
                dout() << "(A) ... setting " << eEndH << ".next = " << eOutH << endl;
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
            dout() << "Case (B) for " << vH << endl;

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

            dout() << "(B) ... setting " << eH << ".next = " << eOutH << endl;
            getE(eH).next = eOutH;
        }
        // --> Case (C): only the outgoing edge is part of a face
        else if (!eIn.face && eOut.face)
        {
            dout() << "Case (C) for " << vH << endl;
            dout() << "(C) ... setting " << eInH << ".next = " << getE(eOut.twin).next << endl;
            eIn.next = getE(eOut.twin).next;
        }
        // --> Case (D): both edges are already part of another face
        if (eIn.face && eOut.face)
        {
            dout() << "Case (D) for " << vH
                   << "(eIn = " << eInH << ", eOut = " << eOutH << ")" << endl;

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

                dout() << "(D) ... setting " << inactiveBladeEndH << ".next = " << getE(eInBladeEndH).next << endl;
                dout() << "(D) ... setting " << eInBladeEndH << ".next = " << getE(eOut.twin).next << endl;
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
    dout() << "+------ Summary face " << newFaceH << " ------+" << endl;
    auto startEdgeH = f.edge;
    auto eH = startEdgeH;
    int i = 0;
    do
    {
        auto& e = getE(eH);
        auto source = getE(e.twin).target;
        auto target = e.target;
        dout() << "| " << source.idx() << " ==> " << eH
               << " ==> " << target.idx() << " [next: " << e.next
               << ", twin-next: " << getE(e.twin).next << "]"
               << endl;

        eH = e.next;
        i++;
        if (i >= 4)
        {
            panic("bug in HEM: face with invalid internal next handles added");
        }
    } while(eH != startEdgeH);
    dout() << "+-----------------------------+" << endl;

    return newFaceH;
}


template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::numVertices() const
{
    return m_vertices.sizeUsed();
}

template <typename BaseVecT>
Point<BaseVecT> HalfEdgeMesh<BaseVecT>::getVertexPosition(VertexHandle handle) const
{
    return getV(handle).pos;
}

template <typename BaseVecT>
size_t HalfEdgeMesh<BaseVecT>::numFaces() const
{
    return m_faces.sizeUsed();
}

template <typename BaseVecT>
std::array<Point<BaseVecT>, 3> HalfEdgeMesh<BaseVecT>::getVertexPositionsOfFace(FaceHandle handle) const
{
    auto handles = getVertexHandlesOfFace(handle);

    auto v1 = getV(handles[0]);
    auto v2 = getV(handles[1]);
    auto v3 = getV(handles[2]);

    return {v1.pos, v2.pos, v3.pos};
}

template <typename BaseVecT>
std::array<VertexHandle, 3>
HalfEdgeMesh<BaseVecT>::getVertexHandlesOfFace(FaceHandle handle) const
{
    auto face = getF(handle);

    auto e1 = getE(face.edge);
    auto e2 = getE(e1.next);
    auto e3 = getE(e2.next);

    return {e1.target, e2.target, e3.target};
}

// ========================================================================
// = Other public methods
// ========================================================================
template <typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::debugCheckMeshIntegrity() const
{
    using std::endl;

    dout() << endl;
    dout() << "===============================" << endl;
    dout() << "===     Integrity check     ===" << endl;
    dout() << "===============================" << endl;

    bool error = false;

    // First: let's visit all faces
    dout() << endl;
    dout() << "+--------------------+" << endl;
    dout() << "| Checking all faces |" << endl;
    dout() << "+--------------------+" << endl;
    for (auto fH : m_faces)
    {
        dout() << "== Checking Face " << fH << "..." << endl;
        auto startEdgeH = getF(fH).edge;
        auto eH = startEdgeH;
        int edgeCount = 0;
        do
        {
            auto& e = getE(eH);
            auto source = getE(e.twin).target;
            auto target = e.target;
            dout() << "   | " << eH << ": " << source << " ==> " << target
                 << " [next: " << e.next << ", twin: " << e.twin
                 << ", twin-face: " << getE(e.twin).face << "]"
                 << endl;

            if (getE(eH).face != fH)
            {
                dout() << "!!!!! Face handle of " << eH << " is " << getE(eH).face
                     << " instead of " << fH << "!!!!!" << endl;
                error = true;
            }

            eH = e.next;
            edgeCount++;
            if (edgeCount >= 20)
            {
                dout() << "   ... stopping iteration after 20 edges." << endl;
            }
        } while(eH != startEdgeH);

        if (edgeCount != 3)
        {
            dout() << "!!!!! More than 3 edges reached from " << fH << endl;
            error = true;
        }
    }

    // Next, we try to reach all boundary edges
    dout() << endl;
    dout() << "+-------------------------------------+" << endl;
    dout() << "| Trying to walk on boundary edges... |" << endl;
    dout() << "+-------------------------------------+" << endl;

    EdgeMap<bool> visited(m_edges.size(), false);
    for (auto startEdgeH : m_edges)
    {
        auto loopEdgeH = startEdgeH;

        if (visited[startEdgeH] || getE(startEdgeH).face)
        {
            continue;
        }
        visited[startEdgeH] = true;

        dout() << "== Starting at " << startEdgeH << endl;

        do
        {
            loopEdgeH = getE(loopEdgeH).next;
            visited[loopEdgeH.idx()] = true;
            dout() << "   | -> " << loopEdgeH
                 << " [twin: " << getE(loopEdgeH).twin << "]" << endl;
        } while(loopEdgeH != startEdgeH);
    }

    // Next, we list all vertices that are not connected to anything yet
    dout() << endl;
    dout() << "+-------------------------------+" << endl;
    dout() << "| List of unconnected vertices: |" << endl;
    dout() << "+-------------------------------+" << endl;

    for (auto vH : m_vertices)
    {
        if (!getV(vH).outgoing)
        {
            dout() << "== " << vH << endl;
        }
    }

    return error;
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
OptionalEdgeHandle
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
        return OptionalEdgeHandle();
    }
}

template <typename BaseVecT>
EdgeHandle
    HalfEdgeMesh<BaseVecT>::findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH)
{
    dout() << "# findOrCreateEdgeBetween: " << fromH << " --> " << toH << endl;
    auto foundEdge = edgeBetween(fromH, toH);
    if (foundEdge)
    {
        dout() << ">> found: " << foundEdge << endl;
        return foundEdge.unwrap();
    }
    else
    {
        dout() << ">> adding pair..." << endl;
        return addEdgePair(fromH, toH).first;
    }
}

template <typename BaseVecT>
template <typename Pred>
OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::findEdgeAroundVertex(VertexHandle vH, Pred pred) const
{
    // This function simply follows `next` and `twin` handles to visit all
    // edges around a vertex.
    dout() << ">> Trying to find an edge around " << vH << " ..." << endl;

    auto& v = getV(vH);
    if (!v.outgoing)
    {
        dout() << ">> ... " << vH << " has no outgoing edge, returning none." << endl;
        return OptionalEdgeHandle();
    }

    return findEdgeAroundVertex(getE(v.outgoing.unwrap()).twin, pred);
}

template <typename BaseVecT>
template <typename Pred>
OptionalEdgeHandle
    HalfEdgeMesh<BaseVecT>::findEdgeAroundVertex(EdgeHandle startEdgeH, Pred pred) const
{
    // This function simply follows `next` and `twin` handles to visit all
    // edges around a vertex.
    dout() << ">> Trying to find an edge starting from " << startEdgeH << " ..." << endl;

    auto loopEdgeH = startEdgeH;

    int iterCount = 0;
    vector<EdgeHandle> visited;

    while (!pred(loopEdgeH))
    {
        dout() << ">> ... >> LOOP: loop-" << loopEdgeH << " @ "
               << getE(loopEdgeH).face
               << " with next: " << getE(loopEdgeH).next << endl;

        loopEdgeH = getE(getE(loopEdgeH).next).twin;
        if (loopEdgeH == startEdgeH)
        {
            dout() << ">> ... we visited all edges once without success, returning none." << endl;
            return OptionalEdgeHandle();
        }

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
    }
    dout() << ">> ... found " << loopEdgeH << "." << endl;
    return loopEdgeH;
}

template <typename BaseVecT>
pair<EdgeHandle, EdgeHandle> HalfEdgeMesh<BaseVecT>::addEdgePair(VertexHandle v1H, VertexHandle v2H)
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
std::ostream& HalfEdgeMesh<BaseVecT>::dout() const
{
    static std::ostringstream fake;

    if (getenv("LVR_MESH_DEBUG") != nullptr)
    {
        return cout;
    }
    else
    {
        // This is a rather hacky solution. We just want to hide the output.
        // Ideally we would write our own "null" implementation of `ostream`,
        // but using a stringstream and deleting its contents from time to time
        // works too.
        fake.str("");
        fake.clear();
        return fake;
    }
}


// ========================================================================
// = Iterator stuff
// ========================================================================

template<typename HandleT>
HemVertexIterator<HandleT>& HemVertexIterator<HandleT>::operator++()
{
    ++m_iterator;
    return *this;
}

template<typename HandleT>
bool HemVertexIterator<HandleT>::operator==(const MeshHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HemVertexIterator<HandleT>*>(&other);
    return cast && m_iterator == cast->m_iterator;
}

template<typename HandleT>
bool HemVertexIterator<HandleT>::operator!=(const MeshHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HemVertexIterator<HandleT>*>(&other);
    return !cast || m_iterator != cast->m_iterator;
}

template<typename HandleT>
HandleT HemVertexIterator<HandleT>::operator*() const
{
    return *m_iterator;
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> HalfEdgeMesh<BaseVecT>::verticesBegin() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<HemVertexIterator<VertexHandle>>(this->m_vertices.begin())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> HalfEdgeMesh<BaseVecT>::verticesEnd() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<HemVertexIterator<VertexHandle>>(this->m_vertices.end())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> HalfEdgeMesh<BaseVecT>::facesBegin() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<HemVertexIterator<FaceHandle>>(this->m_faces.begin())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> HalfEdgeMesh<BaseVecT>::facesEnd() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<HemVertexIterator<FaceHandle>>(this->m_faces.end())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> HalfEdgeMesh<BaseVecT>::edgesBegin() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<HemVertexIterator<EdgeHandle>>(this->m_edges.begin())
    );
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> HalfEdgeMesh<BaseVecT>::edgesEnd() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<HemVertexIterator<EdgeHandle>>(this->m_edges.end())
    );
}

} // namespace lvr

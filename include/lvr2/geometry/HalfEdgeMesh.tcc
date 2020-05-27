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
 * HalfEdgeMesh.tcc
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <algorithm>
#include <array>
#include <utility>
#include <iostream>

#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/util/Panic.hpp"
#include "lvr2/util/Debug.hpp"


namespace lvr2
{

template<typename BaseVecT>
HalfEdgeMesh<BaseVecT>::HalfEdgeMesh()
{
}

template<typename BaseVecT>
HalfEdgeMesh<BaseVecT>::HalfEdgeMesh(MeshBufferPtr ptr)
{
    size_t numFaces = ptr->numFaces();
    size_t numVertices = ptr->numVertices();

    floatArr vertices = ptr->getVertices();
    indexArray indices = ptr->getFaceIndices();

    for(size_t i = 0; i < numVertices; i++)
    {
        size_t pos = 3 * i;
        this->addVertex(BaseVecT(
                            vertices[pos],
                            vertices[pos + 1],
                            vertices[pos + 2]));
    }

    for(size_t i = 0; i < numFaces; i++)
    {
        size_t pos = 3 * i;
        VertexHandle v1(indices[pos]);
        VertexHandle v2(indices[pos + 1]);
        VertexHandle v3(indices[pos + 2]);
        try
        {
            this->addFace(v1, v2, v3);
        }
        catch(PanicException exception)
        {
            std::cerr << timestamp << "Warning loop detected. Omitting face "
                      << v1.idx() << " " << v2.idx() << " " << v3.idx() << std::endl;
        }
    }
}


// ========================================================================
// = Interface methods
// ========================================================================

template <typename BaseVecT>
VertexHandle HalfEdgeMesh<BaseVecT>::addVertex(BaseVecT pos)
{
    Vertex v;
    v.pos = pos;
    return m_vertices.push(v);
}

template <typename BaseVecT>
FaceHandle HalfEdgeMesh<BaseVecT>::addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H)
{
    using std::make_tuple;

    DOINDEBUG(dout() << "##################################################" << endl);
    DOINDEBUG(dout() << "##### addFace(): " << v1H << " -> " << v2H << " -> " << v3H << endl);

    // =======================================================================
    // = Create broken edges
    // =======================================================================
    // Handles for the inner edges of the face. The edges represented by those
    // handles do not contain a valid `next` pointer yet.

    bool addedE1, addedE2, addedE3;

    auto eInner1H = findOrCreateEdgeBetween(v1H, v2H, addedE1);
    auto eInner2H = findOrCreateEdgeBetween(v2H, v3H, addedE2);
    auto eInner3H = findOrCreateEdgeBetween(v3H, v1H, addedE3);

    auto &eInner1 = getE(eInner1H);
    auto &eInner2 = getE(eInner2H);
    auto &eInner3 = getE(eInner3H);

    auto eOuter1H = eInner1.twin;
    auto eOuter2H = eInner2.twin;
    auto eOuter3H = eInner3.twin;

    auto &v1 = getV(v1H);
    auto &v2 = getV(v2H);
    auto &v3 = getV(v3H);

    // =======================================================================
    // = Create face
    // =======================================================================
    FaceHandle newFaceH = m_faces.nextHandle();
    Face f(eInner1H);
    m_faces.push(f);

    auto old_faces = make_tuple(eInner1.face, eInner2.face, eInner3.face);

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
        make_tuple(eOuter3H, v3H, eOuter2H)};

    std::map<Index, Index> previousNexts;

    try{
    for (auto corner : corners)
    {
        auto eInH = get<0>(corner);
        auto vH = get<1>(corner);
        auto eOutH = get<2>(corner);

        auto &eIn = getE(eInH);
        auto &v = getV(vH);
        auto &eOut = getE(eOutH);

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
                auto eEndH = findEdgeAroundVertex(vH, [this](auto edgeH) {
                                 return !getE(edgeH).face;
                             })
                                 .unwrap();

                if (previousNexts.find(eInH.idx()) == previousNexts.end())
                {
                    previousNexts[eInH.idx()] = eIn.next.idx();
                }
                if (previousNexts.find(eEndH.idx()) == previousNexts.end())
                {
                    previousNexts[eEndH.idx()] = getE(eEndH).next.idx();
                }

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
            auto eHOpt = findEdgeAroundVertex(vH, [&, this](auto edgeH) {
                          auto oldNext = eIn.twin;
                          return !getE(edgeH).face && getE(edgeH).next == oldNext;
                      });

            auto eH = eHOpt.unwrap();
            
            DOINDEBUG(dout() << "(B) ... setting " << eH << ".next = " << eOutH << endl);

            if (previousNexts.find(eH.idx()) == previousNexts.end())
            {
                previousNexts[eH.idx()] = getE(eH).next.idx();
            }

            getE(eH).next = eOutH;
        }
        // --> Case (C): only the outgoing edge is part of a face
        else if (!eIn.face && eOut.face)
        {
            DOINDEBUG(dout() << "Case (C) for " << vH << endl);
            DOINDEBUG(
                dout() << "(C) ... setting " << eInH << ".next = "
                       << getE(eOut.twin).next << endl);

            previousNexts[eInH.idx()] = eIn.next.idx();

            eIn.next = getE(eOut.twin).next;
        }
        // --> Case (D): both edges are already part of another face
        if (eIn.face && eOut.face)
        {
            DOINDEBUG(
                dout() << "Case (D) for " << vH
                       << "(eIn = " << eInH << ", eOut = " << eOutH << ")" << endl);

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
                auto inactiveBladeEndH = findEdgeAroundVertex(vH, [&, this](auto edgeH) {
                                             return getE(edgeH).next == eIn.twin;
                                         })
                                             .unwrap();

                // Instead of pointing to `eIn.twin`, it needs to "skip" the
                // `eIn` blade and point to the blade afterwards. So we need to
                // find the end of the `eIn` blade. Its next handle is the one
                // `inactiveBladeEnd` needs to point to.
                auto eInBladeEndH = findEdgeAroundVertex(
                                        eInH,
                                        [&, this](auto edgeH) {
                                            return !getE(edgeH).face;
                                        })
                                        .unwrap();

                if (previousNexts.find(inactiveBladeEndH.idx()) == previousNexts.end())
                {
                    previousNexts[inactiveBladeEndH.idx()] = getE(inactiveBladeEndH).next.idx();
                }
                if (previousNexts.find(eInBladeEndH.idx()) == previousNexts.end())
                {
                    previousNexts[eInBladeEndH.idx()] = getE(eInBladeEndH).next.idx();
                }

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
                           << getE(eInBladeEndH).next << endl);
                DOINDEBUG(
                    dout() << "(D) ... setting " << eInBladeEndH << ".next = "
                           << getE(eOut.twin).next << endl);
            }
        }
    }

    }
    catch(PanicException e)
    {
      for (auto it = previousNexts.begin(); it != previousNexts.end(); ++it)
      {
        getE(HalfEdgeHandle(it->first)).next = HalfEdgeHandle(it->second);
      }

      m_faces.erase(newFaceH);

      if(addedE1)
      {
        m_edges.erase(eOuter1H);
        m_edges.erase(eInner1H);
      }
      else {
        eInner1.face = get<0>(old_faces);
      }
      if(addedE2)
      {
        m_edges.erase(eOuter2H);
        m_edges.erase(eInner2H);
      }
      else {
        eInner2.face = get<1>(old_faces);
      }
      if(addedE3)
      {
        m_edges.erase(eOuter3H);
        m_edges.erase(eInner3H);
      }
      else {
        eInner3.face = get<2>(old_faces);
      }

      // propagate the error up.
      throw e;
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
        auto &e = getE(eH);
        auto source = getE(e.twin).target;
        auto target = e.target;
        DOINDEBUG(
            dout() << "| " << source.idx() << " ==> " << eH
                   << " ==> " << target.idx() << " [next: " << e.next
                   << ", twin-next: " << getE(e.twin).next << "]"
                   << endl);

        eH = e.next;
        i++;
        if (i >= 4)
        {
            panic("bug in HEM: face with invalid internal next handles added");
        }
    } while (eH != startEdgeH);
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
BaseVecT HalfEdgeMesh<BaseVecT>::getVertexPosition(VertexHandle handle) const
{
    return getV(handle).pos;
}

template <typename BaseVecT>
BaseVecT& HalfEdgeMesh<BaseVecT>::getVertexPosition(VertexHandle handle)
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
OptionalFaceHandle HalfEdgeMesh<BaseVecT>::getOppositeFace(FaceHandle faceH, VertexHandle vertexH) const
{
  auto e = getE(getF(faceH).edge);
  for(size_t i=0; i<3; i++)
  {
    auto next = getE(e.next);
    if(next.target == vertexH)
      return getE(e.twin).face;
    e = next;
  }
  return OptionalFaceHandle();
}

template <typename BaseVecT>
OptionalEdgeHandle HalfEdgeMesh<BaseVecT>::getOppositeEdge(FaceHandle faceH, VertexHandle vertexH) const
{
  auto eH = getF(faceH).edge;
  for(size_t i=0; i<3; i++)
  {
    auto nH = getE(eH).next;
    if(getE(nH).target == vertexH)
      return halfToFullEdgeHandle(eH);
    eH = nH;
  }
  return OptionalEdgeHandle();
}

template <typename BaseVecT>
OptionalVertexHandle HalfEdgeMesh<BaseVecT>::getOppositeVertex(FaceHandle faceH, EdgeHandle edgeH) const
{
  auto e1 = getE(HalfEdgeHandle::oneHalfOf(edgeH));
  if(e1.face && e1.face.unwrap() == faceH)
    return getE(e1.next).target;

  auto e2 = getE(e1.twin);
  if(e2.face && e2.face.unwrap() == faceH)
    return getE(e2.next).target;

  else
    return OptionalVertexHandle();
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

template<typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::isBorderEdge(EdgeHandle handle) const
{
    HalfEdgeHandle h = HalfEdgeHandle::oneHalfOf(handle);

    auto edge = getE(h);
    auto twin = getE(edge.twin);

    // Only if both halfs have an adjacent face this is not a border edge
    if(twin.face && edge.face)
    {
        return false;
    }

    return true;
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

        // Throw an exception if number of out edges becomes
        // too large. This can happen if there is a bug in the
        // half edge mesh topology
        if(edgesOut.size() > 40)
        {
            throw VertexLoopException("getEdgesOfVertex: Loop detected");
        }
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

/**
 * finds the common neighbors of two vertices
 * @tparam BaseVecT
 * @param vH1
 * @param vH2
 * @return a vector with the common neighbors
 */
template <typename BaseVecT>
vector<VertexHandle> HalfEdgeMesh<BaseVecT>::findCommonNeigbours(VertexHandle vH1, VertexHandle vH2){
    vector<VertexHandle> vH1nb = this->getNeighboursOfVertex(vH1);
    vector<VertexHandle> vH2nb = this->getNeighboursOfVertex(vH2);

    vector<VertexHandle> commonVertexHandles;

    for(auto i = vH1nb.begin(); i != vH1nb.end(); ++i)
    {
        if(find(vH2nb.begin(), vH2nb.end(), *i) != vH2nb.end())
        {
            commonVertexHandles.push_back(*i);
        }
    }

    return commonVertexHandles;
}



/**
 * Calculates the triangle circumcircle and circumcenter.
 * Triangle circumcenter in R^3:
 *
 *             |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)]
 *     m = a + ---------------------------------------------------------.
 *                                 2 | (b-a)x(c-a) |^2
 * @tparam BaseVecT
 * @param faceH
 * @return
 */
template <typename BaseVecT>
std::pair<BaseVecT, float> HalfEdgeMesh<BaseVecT>::triCircumCenter(FaceHandle faceH) {
    //get vertices of the face
    auto vertices = getVerticesOfFace(faceH);
    BaseVecT a = getV(vertices[0]).pos;
    BaseVecT b = getV(vertices[1]).pos;
    BaseVecT c = getV(vertices[2]).pos;

    BaseVecT circumCenter = a;

    float radius;

    BaseVecT cMinusA = c-a;
    BaseVecT bMinusA = b-a;

    BaseVecT numerator = ( (bMinusA.cross(cMinusA).cross(bMinusA)) * cMinusA.length2()  +  (cMinusA.cross(bMinusA.cross(cMinusA))) * bMinusA.length2() );
    float denominator = ( 2 * (bMinusA.cross(cMinusA)).length2());

    circumCenter += numerator / denominator;
    radius = (circumCenter-a).length();


    return std::make_pair(circumCenter, radius);

}

/**
 * performs an edge split operation on the given edge
 * @tparam BaseVecT
 * @param edgeH
 * @return a result containg the newly added vertex and the new faces
 */
template <typename BaseVecT>
EdgeSplitResult HalfEdgeMesh<BaseVecT>::splitEdge(EdgeHandle edgeH) {

    if(this->isBorderEdge(edgeH))
    {
        std::cout << "splitEdge() cannot be called with border edge" << endl;
        VertexHandle dummy(std::numeric_limits<int>::max());
        return EdgeSplitResult(dummy); //return a result with a dummy vector
    }
    // A fancy drawing of the current and expected situation:
    //
    //                                 |                                     |
    //            Before               |                After                |
    //            ------               |                -----                |
    //                                 |                                     |
    //                                 |                                     |
    //              [C]                |                 [C]                 |
    //                                 |                                     |
    //          ^  /   ^  \            |             ^  /   ^  \             |
    //         /  / ^ | \  \           |            /  / ^ | \  \            |
    //        /  /  | |  \  \          |           /  /  | |  \  \           |
    //       /  /   | |   \  \         |          /  /   | |   \  \          |
    //      /  /c   | |   d\  \        |         /  /    | |    \  \         |
    //     /  /     | |     \  \       |        /  /     | |     \  \        |
    //    /  /      | |      \  \      |       /  /      | |      \  \       |
    //   /  /       | |       \  \     |      /  /       | |       \  \      |
    //  /  v        | |        \  v    |     /  v        | v        \  v     |
    //              | |                |          ----->     ----->          |
    //  [A]  (X)    a  b   (Y)  [B]    |     [A]         [E]         [B]     |
    //              | |                |          <-----     <-----          |
    //  ^  \        | |        ^  /    |     ^  \        ^ |        ^  /     |
    //   \  \       | |       /  /     |      \  \       | |       /  /      |
    //    \  \      | |      /  /      |       \  \      | |      /  /       |
    //     \  \     | |     /  /       |        \  \     | |     /  /        |
    //      \  \e   | |   f/  /        |         \  \    | |    /  /         |
    //       \  \   | |   /  /         |          \  \   | |   /  /          |
    //        \  \  | |  /  /          |           \  \  | |  /  /           |
    //         \  \ | v /  /           |            \  \ | v /  /            |
    //          \  v   /  v            |             \  v   /  v             |
    //                                 |                                     |
    //              [D]                |                 [D]                 |
    //
    //
    // ### A mapping from graphic names to variable names:
    //
    //  Edges                      Vertices               Faces
    //  -----                      --------               -----
    //  a: center                  [A]: vLeft             (X): faceLeft
    //  b: centerTwin              [B]: vRight            (Y): faceRight
    //  c: aboveLeft               [C]: vAbove
    //  d: aboveRight              [D]: vBelow
    //  e: belowLeft               [E]: vAdded
    //  f: belowRight
    //
    //
    // We just imagine that the random half-edge we get from `oneHalfOf()` is
    // the edge `a` in the drawing. We call it `center`.
    //
    // First, we just obtain all the handles

    HalfEdgeHandle centerH = HalfEdgeHandle::oneHalfOf(edgeH); // 1 (X)
    HalfEdge& center = getE(centerH);
    HalfEdgeHandle centerTwinH = getE(centerH).twin; //e2 (target->vM, next->e9)
    HalfEdge& centerTwin = getE(centerTwinH);

    //get the two faces
    FaceHandle topLeftH = getE(centerH).face.unwrap(); //f1 (edge->e1)
    FaceHandle topRightH = getE(centerTwinH).face.unwrap();  //f2 (edge->e9)

    //get missing halfedges
    HalfEdgeHandle aboveLeftH = getE(centerH).next; //e3 (next->e7)
    HalfEdge& aboveLeft = getE(aboveLeftH);

    HalfEdgeHandle belowRightH = getE(centerTwinH).next; //e5 (next->e10, face->f4)
    HalfEdge& belowRight = getE(belowRightH);

    HalfEdgeHandle aboveRightH = getE(belowRightH).next; //e4 (next->e11, face->f3)
    HalfEdge& aboveRight = getE(aboveRightH);

    HalfEdgeHandle belowLeftH = getE(aboveLeftH).next; //e6 (X)
    HalfEdge& belowLeft = getE(belowLeftH);

    //get vertices
    VertexHandle vLeftH = aboveLeft.target; //vL (X)
    VertexHandle vRightH = belowRight.target; //vR (X)
    VertexHandle vAboveH = center.target; //vO (X)
    VertexHandle vBelowH = centerTwin.target; //vU (outgoing -> e11)

    //calculate the new vertex
    BaseVecT vAddedV = getV(vAboveH).pos + (getV(vBelowH).pos - getV(vAboveH).pos) / 2;

    //add it
    VertexHandle vAddedH = this->addVertex(vAddedV); //vM (outgoing -> e9)
    Vertex& vAdded = getV(vAddedH);


    //CREATE 6 NEW HALFEDGES
    HalfEdgeHandle leftAddedH = findOrCreateEdgeBetween(vLeftH, vAddedH); //e7 (face->f1, next->e1)
    HalfEdgeHandle addedLeftH = getE(leftAddedH).twin; //e8 (face->f3, next->e4)

    HalfEdgeHandle rightAddedH = findOrCreateEdgeBetween(vRightH, vAddedH); //e10 (face->f4, next->e12)
    HalfEdgeHandle addedRightH = getE(rightAddedH).twin; //e9 (face->f2, next->e6)

    HalfEdgeHandle belowAddedH = findOrCreateEdgeBetween(vBelowH, vAddedH); //e11 (face->f3, next->e8)
    HalfEdgeHandle addedBelowH = getE(belowAddedH).twin; //e12 (face->f4, next->e5)

    //FIX OUTGOINGS, WHICH MIGHT BE BROKEN
    getV(vBelowH).outgoing = belowAddedH;
    getV(vAddedH).outgoing = addedRightH;

    Face bottomLeft(addedBelowH);
    FaceHandle bottomLeftH = m_faces.push(bottomLeft);
    getF(bottomLeftH).edge = belowAddedH;

    Face bottomRight(rightAddedH);
    FaceHandle bottomRightH = m_faces.push(bottomRight);
    getF(bottomRightH).edge = rightAddedH;

    //SET EDGES OF UPPER FACES
    getF(topRightH).edge = addedRightH;
    getF(topLeftH).edge = centerH;

    //fix center handles
    //getE(centerH).next = aboveLeftH;
    getE(centerTwinH).next = addedRightH;

    //getE(centerH).target = vAboveH;
    getE(centerTwinH).target = vAddedH;

    //we need to redirect all the Halfedges
    getE(leftAddedH).next = centerH;
    getE(addedLeftH).next = belowLeftH;

    getE(rightAddedH).next = addedBelowH;
    getE(addedRightH).next = aboveRightH;

    getE(belowAddedH).next = addedLeftH;
    getE(addedBelowH).next = belowRightH;

    getE(aboveLeftH).next = leftAddedH;
    //getE(aboveRightH).next = centerTwinH;
    getE(belowLeftH).next = belowAddedH;
    getE(belowRightH).next = rightAddedH;

    //now, that all the edges are redirected, we need to insert new faces (4(2?)) and set the faces of each inner edge

    getE(leftAddedH).face = topLeftH;
    getE(addedLeftH).face = bottomLeftH;

    getE(rightAddedH).face = bottomRightH;
    getE(addedRightH).face = topRightH;

    getE(belowAddedH).face = bottomLeftH;
    getE(addedBelowH).face = bottomRightH;

    getE(belowLeftH).face = bottomLeftH;
    getE(belowRightH).face = bottomRightH;

    getE(centerH).face = topLeftH;
    getE(centerTwinH).face = topRightH;

    //fill edge split result
    EdgeSplitResult result(vAddedH);
    result.addedFaces.push_back(topLeftH);
    result.addedFaces.push_back(topRightH);
    result.addedFaces.push_back(bottomLeftH);
    result.addedFaces.push_back(bottomRightH);

    return result;
}

/**
 * performs a vertex split operation on the selected vertex
 * @tparam BaseVecT
 * @param vertexToBeSplitH
 * @return a struct containing the new vertex and the added faces
 */
template <typename BaseVecT>
VertexSplitResult HalfEdgeMesh<BaseVecT>::splitVertex(VertexHandle vertexToBeSplitH)
{

    HalfEdge longestOutgoingEdge;
    BaseVecT vertexToBeSplit = getV(vertexToBeSplitH).pos;
    auto outEdges = getEdgesOfVertex(vertexToBeSplitH);

    float longestDistance = 0; //save length of longest edge
    EdgeHandle longestEdge(0); //save longest edge
    HalfEdge longestEdgeHalf; //needed for vertex calc

    /************************************
     * Get vertex, which will be added. *
     ************************************/

    // determine longest outgoing edge
    for(EdgeHandle edge : outEdges)
    {
        HalfEdgeHandle halfH = HalfEdgeHandle::oneHalfOf(edge); //get Halfedge
        HalfEdge half = getE(halfH);

        //make halfedge direct to the target of the longest edge
        if(half.target == vertexToBeSplitH)
        {
            halfH = half.twin;
            half = getE(halfH);
        }

        BaseVecT target = getV(half.target).pos;
        auto distance = target.distanceFrom(getV(vertexToBeSplitH).pos);
        //changes values to longer edge
        if(distance > longestDistance){
            longestDistance = distance;
            longestEdge = edge;
            longestEdgeHalf = half;
        }
    }

    VertexHandle targetOfLongestEdgeH = longestEdgeHalf.target;
    BaseVecT targetOfLongestEdge = getV(targetOfLongestEdgeH).pos;

    //do an edge split on the longest edge and do an edge flip for each of the 2 found vertices
    //if the local delaunay criteria is not met
    vector<VertexHandle> commonVertexHandles = findCommonNeigbours(vertexToBeSplitH, targetOfLongestEdgeH);

    EdgeSplitResult splitResult = this->splitEdge(longestEdge);


    //only flip if a criteria is given (local delaunay criteria)
    for(VertexHandle vertex : commonVertexHandles)
    {
        OptionalEdgeHandle handle = this->getEdgeBetween(vertex,vertexToBeSplitH);
        if(handle)
        {
            auto faces = this->getFacesOfEdge(handle.unwrap());
            FaceHandle fH1 = faces[0].unwrap();
            FaceHandle fH2 = faces[1].unwrap();

            auto f1Vertices = getVerticesOfFace(fH1);
            auto f2Vertices = getVerticesOfFace(fH2);

            //calculate the circumcenter of each triangle, look whether the local delaunay criteria is fulfilled
            auto circumCenterPair1 = triCircumCenter(fH1);
            auto circumCenterPair2 = triCircumCenter(fH2);

            BaseVecT circumCenter1 = circumCenterPair1.first;
            BaseVecT circumCenter2 = circumCenterPair2.first;

            float radius1 = circumCenterPair1.second;
            float radius2 = circumCenterPair2.second;

            BaseVecT singleFace1;
            BaseVecT singleFace2;

            //get the vertices not part of the center edge
            for(int i = 0; i< 3; i++)
            {
                VertexHandle current = f1Vertices[i];
                if(current != vertex && current != vertexToBeSplitH) singleFace1 = getV(current).pos;
                current = f2Vertices[i];
                if(current != vertex && current != vertexToBeSplitH) singleFace2 = getV(current).pos;
            }

            //flip only, if one of the single vertices is inside the circumcircle of the other triangle
            if((singleFace1-circumCenter2).length() <= radius2 || (singleFace2-circumCenter1).length() <= radius1)
            {
                if(this->isFlippable(handle.unwrap()))
                {
                    this->flipEdge(handle.unwrap());
                }
            }
        }
    }


    VertexSplitResult result(splitResult.edgeCenter);
    result.addedFaces.assign(splitResult.addedFaces.begin(), splitResult.addedFaces.end());

    return result;
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
    EdgeCollapseResult result(vertexToKeepH, vertexToRemoveH);

    // Fix targets for ingoing edges of the vertex that will be deleted. This
    // has to be done before changing the twin edges.
    circulateAroundVertex(vertexToRemoveH, [&, this](auto ingoingEdgeH)
    {
        getE(ingoingEdgeH).target = vertexToKeepH;
        return true;
    });

    // Save edges to delete for later
    boost::optional<array<HalfEdgeHandle, 2>> edgesToDeleteAbove;
    boost::optional<array<HalfEdgeHandle, 2>> edgesToDeleteBelow;

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

/**
 * Checks whether the given edge is flippable or not. includes check for huetchen structures
 * @tparam BaseVecT
 * @param handle
 * @return if the edge is flippable
 */
template<typename BaseVecT>
bool HalfEdgeMesh<BaseVecT>::isFlippable(EdgeHandle handle) const
{
    auto adjFaces = getFacesOfEdge(handle);
    if (!adjFaces[0] || !adjFaces[1])
    {
        return false;
    }

    //check huetchen
    HalfEdgeHandle hEH = HalfEdgeHandle::oneHalfOf(handle);
    auto target1 = getE(hEH).target;
    auto target2 = getE(getE(hEH).twin).target;

    int count1 = getEdgesOfVertex(target1).size();
    int count2 = getEdgesOfVertex(target2).size();

    if(count1 <= 5 || count2 <= 5) return false; //4 or 5?


    //works only for huetchens which are not connected to any other structure

    if(getE(hEH).face && getE(getE(getE(hEH).next).twin).face && getE(getE(getE(getE(hEH).next).next).twin).face)
    {
        if(getE(getE(getE(getE(hEH).next).twin).next).next.idx() == getE(getE(getE(getE(getE(hEH).next).next).twin).next).twin.idx())
        {
            cout << "Hütchen detected" << endl;
            return false;
        }
    }



    // Make sure we have 4 different vertices around the faces of that edge.
    auto faces = getFacesOfEdge(handle);
    auto face0vertices = getVerticesOfFace(faces[0].unwrap());
    auto face1vertices = getVerticesOfFace(faces[1].unwrap());

    auto diffCount = std::count_if(face0vertices.begin(), face0vertices.end(), [&](auto f0v)
    {
        return std::find(face1vertices.begin(), face1vertices.end(), f0v) == face1vertices.end();
    });
    return diffCount == 1;
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

    if(center.face && centerTwin.face)
    {
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
    else
    {
        //cout << "Edge not flippable!" << endl;
        return;
    }
}

template <typename BaseVecT>
void HalfEdgeMesh<BaseVecT>::splitVertex(EdgeHandle eH,
                                         VertexHandle vH,
                                         BaseVecT pos1,
                                         BaseVecT pos2)
{
    // A fancy drawing of the current and expected situation:
    //
    //                                     |                                     |
    //               Before                |                After                |
    //               ------                |                -----                |
    //                                     |                                     |
    //                                     |                                     |
    //                 [ ]                 |                 [ ]                 |
    //                                     |                                     |
    //             ^  /   ^  \             |             ^  /   ^  \             |
    //            /  / ^ | \  \            |            /  / ^ | \  \            |
    //           /  /  | |  \  \           |           /  /  | |  \  \           |
    //          /  /   | |   \  \          |          /  /   | |   \  \          |
    //         /  /    | |    \  \         |         /  /    | |    \  \         |
    //        /  /     | |     \  \        |        /  /   f | | e   \  \        |
    //       /  /      | |      \  \       |       /  /      | |      \  \       |
    //      /  /       | |       \  \      |      /  /       | |       \  \      |
    //     /  v        | |        \  v     |     /  v   a    | v    d   \  v     |
    //                 eH                  |         ------>     ------>         |
    //     [ ]         | |         [ ]     |     [ ]     [newVertex]     [ ]     |
    //                 | |                 |         <------     <------         |
    //     ^  \        | |        ^  /     |     ^  \        ^ |        ^  /     |
    //      \  \v1Edge | |       /  /      |      \  \       | |       /  /      |
    //       \  \      | |      /  /       |       \  \ (X)  | |  (Y) /  /       |
    //        \  \     | |     /  /        |        \  \     | |     /  /        |
    //         \  \    | |    /  /         |       b \  \    | |    /  / c       |
    //          \  \   | |   /  /          |          \  \   | |   /  /          |
    //           \  \  | |  /  /v2Edge     |           \  \  | |  /  /           |
    //            \  \ | v /  /            |            \  \ | v /  /            |
    //             \  v   /  v             |             \  v   /  v             |
    //                                     |                                     |
    //                 [vH]                |                 [vH]                |
    //
    // The position of [vH] will be set to pos1 and the position of [newVertex] to pos2.
    // Additionally the Faces (X) and (Y) will be added in between of v1Edge and v1Edge.twin or
    // v2Edge and v2Edge.twin. So the halfEdge a was v1Edge before, b was v1Edge.twin, c was v2Edge
    // and d was v2Edge.twin. Also the halfEdges e and f represent the edge that was given by eH
    // initially.

    HalfEdgeHandle halfEdge = HalfEdgeHandle::oneHalfOf(eH);
    if (getE(halfEdge).target == vH)
    {
        halfEdge = getE(halfEdge).twin;
    }

    HalfEdgeHandle v1Edge = getE(getE(halfEdge).next).next;
    HalfEdgeHandle v2Edge = getE(getE(getE(halfEdge).twin).next).twin;

    // find edges that will point at the new vertex
    std::vector<HalfEdgeHandle> edges;
    circulateAroundVertex(vH, [v1Edge, v2Edge, &edges](auto eH) {
        if (eH == v1Edge)
        {
            edges.clear();
        }
        if (eH == v2Edge)
        {
            return false;
        }

        edges.push_back(eH);

        return true;
    });

    // create new vertex and edit vertex positions
    getV(vH).pos           = pos1;
    VertexHandle newVertex = addVertex(pos2);

    // make all edges that point to vH from one side of the vertex reference the newly created
    // vewVertex
    for (HalfEdgeHandle eH : edges)
    {
        getE(eH).target = newVertex;
    }

    // create new edges
    pair<HalfEdgeHandle, HalfEdgeHandle> newEdge1
        = addEdgePair(newVertex, getE(getE(v1Edge).twin).target);
    pair<HalfEdgeHandle, HalfEdgeHandle> newEdge2 = addEdgePair(getE(getE(v2Edge).twin).target, vH);
    pair<HalfEdgeHandle, HalfEdgeHandle> newEdgeC = addEdgePair(vH, newVertex);

    // configure new edges for first split edge
    getE(newEdge1.first).next  = newEdgeC.second;
    getE(newEdgeC.second).next = newEdge1.second;
    getE(newEdge1.second).next = newEdge1.first;

    // change twins for split edge
    getE(newEdge1.first).twin    = getE(v1Edge).twin;
    getE(getE(v1Edge).twin).twin = newEdge1.first;

    getE(newEdge1.second).twin = v1Edge;
    getE(v1Edge).twin          = newEdge1.second;

    // configure new edges for second split edge
    getE(newEdge2.first).next  = newEdge2.second;
    getE(newEdge2.second).next = newEdgeC.first;
    getE(newEdgeC.first).next  = newEdge2.first;

    // change twins for split edge
    getE(newEdge2.second).twin   = getE(v2Edge).twin;
    getE(getE(v2Edge).twin).twin = newEdge2.second;

    getE(newEdge2.first).twin = v2Edge;
    getE(v2Edge).twin         = newEdge2.first;

    // configure vertices to be consistent with new edges
    getV(vH).outgoing        = getE(v1Edge).twin;
    getV(newVertex).outgoing = newEdge1.second;

    // create a new face for each side of the split vertex
    Face newFace1(newEdgeC.second);

    FaceHandle newFace1H = m_faces.nextHandle();
    m_faces.push(newFace1);

    getE(newEdge1.first).face  = newFace1H;
    getE(newEdge1.second).face = newFace1H;
    getE(newEdgeC.second).face = newFace1H;

    Face newFace2(newEdgeC.first);

    FaceHandle newFace2H = m_faces.nextHandle();
    m_faces.push(newFace2);

    getE(newEdge2.first).face  = newFace2H;
    getE(newEdge2.second).face = newFace2H;
    getE(newEdgeC.first).face  = newFace2H;
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
        auto eH         = startEdgeH;
        int edgeCount   = 0;
        do
        {
            auto& e     = getE(eH);
            auto source = getE(e.twin).target;
            auto target = e.target;
            cout << "   | " << eH << ": " << source << " ==> " << target << " [next: " << e.next
                 << ", twin: " << e.twin << ", twin-face: " << getE(e.twin).face << "]" << endl;

            if (getE(eH).face != fH)
            {
                cout << "!!!!! Face handle of " << eH << " is " << getE(eH).face << " instead of "
                     << fH << "!!!!!" << endl;
                error = true;
            }

            eH = e.next;
            edgeCount++;
            if (edgeCount >= 20)
            {
                cout << "   ... stopping iteration after 20 edges." << endl;
                break;
            }
        } while (eH != startEdgeH);

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
            loopEdgeH          = getE(loopEdgeH).next;
            const auto twinH   = getE(loopEdgeH).twin;
            visited[loopEdgeH] = true;
            cout << "   | -> " << loopEdgeH << " [twin: " << twinH << " | " << getE(twinH).target
                 << " --> " << getE(loopEdgeH).target << "]" << endl;
        } while (loopEdgeH != startEdgeH);
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
HalfEdgeHandle
HalfEdgeMesh<BaseVecT>::findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH, bool& added)
{
  DOINDEBUG(dout() << "# findOrCreateEdgeBetween: " << fromH << " --> " << toH << endl);
  auto foundEdge = edgeBetween(fromH, toH);
  if (foundEdge)
  {
    DOINDEBUG(dout() << ">> found: " << foundEdge << endl);
    added = false;
    return foundEdge.unwrap();
  }
  else
  {
    DOINDEBUG(dout() << ">> adding pair..." << endl);
    added = true;
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

    int iterCount = 0;
    vector<HalfEdgeHandle> visited;

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

        iterCount++;
        if (iterCount > 100)
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

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
 * PmpMesh.tcc
 *
 *  @date 06.12.2021
 *  @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "lvr2/algorithm/pmp/SurfaceHoleFilling.h"
#include "lvr2/algorithm/pmp/SurfaceSmoothing.h"
#include "lvr2/util/Progress.hpp"

#include <unordered_set>

namespace lvr2
{

template<typename BaseVecT>
PmpMesh<BaseVecT>::PmpMesh(MeshBufferPtr ptr)
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
        this->addFace(v1, v2, v3);
    }
}

template<typename BaseVecT>
EdgeCollapseResult PmpMesh<BaseVecT>::collapseEdge(EdgeHandle edgeH)
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    if (!m_mesh.is_collapse_ok(heH))
    {
        panic("call to collapseEdge() with non-collapsable edge!");
    }
    pmp::Halfedge opH = m_mesh.opposite_halfedge(heH);

    Vertex keptH = m_mesh.to_vertex(heH);
    Vertex removedH = m_mesh.from_vertex(heH);
    pmp::Point& pos = m_mesh.position(keptH);
    pos += (m_mesh.position(removedH) - pos) / 2.0;

    EdgeCollapseResult result(keptH, removedH);

    Face f0 = m_mesh.face(heH);
    Face f1 = m_mesh.face(opH);
    if (f0.is_valid())
    {
        result.neighbors[0] = EdgeCollapseRemovedFace(
            f0,
            {
                m_mesh.edge(m_mesh.prev_halfedge(heH)),
                m_mesh.edge(m_mesh.next_halfedge(heH))
            },
            m_mesh.edge(m_mesh.next_halfedge(heH))
        );
    }
    if (f1.is_valid())
    {
        result.neighbors[1] = EdgeCollapseRemovedFace(
            f1,
            {
                m_mesh.edge(m_mesh.next_halfedge(opH)),
                m_mesh.edge(m_mesh.prev_halfedge(opH))
            },
            m_mesh.edge(m_mesh.prev_halfedge(opH))
        );
    }

    m_mesh.collapse(heH);
    return result;
}

template<typename BaseVecT>
array<VertexHandle, 3> PmpMesh<BaseVecT>::getVerticesOfFace(FaceHandle handle) const
{
    auto f = m_mesh.vertices(handle);
    array<VertexHandle, 3> result = { *f, *(++f), *(++f) };
    return result;
}
template<typename BaseVecT>
array<EdgeHandle, 3> PmpMesh<BaseVecT>::getEdgesOfFace(FaceHandle handle) const
{
    auto f = m_mesh.halfedges(handle);
    array<EdgeHandle, 3> result = { m_mesh.edge(*f), m_mesh.edge(*(++f)), m_mesh.edge(*(++f)) };
    return result;
}
template<typename BaseVecT>
void PmpMesh<BaseVecT>::getNeighboursOfFace(FaceHandle handle, vector<FaceHandle>& facesOut) const
{
    for (pmp::Halfedge heH : m_mesh.halfedges(handle))
    {
        FaceHandle face = m_mesh.face(m_mesh.opposite_halfedge(heH));
        if (face.is_valid())
        {
            facesOut.push_back(face);
        }
    }
}
template<typename BaseVecT>
array<VertexHandle, 2> PmpMesh<BaseVecT>::getVerticesOfEdge(EdgeHandle edgeH) const
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    return { m_mesh.to_vertex(heH), m_mesh.from_vertex(heH) };
}
template<typename BaseVecT>
array<OptionalFaceHandle, 2> PmpMesh<BaseVecT>::getFacesOfEdge(EdgeHandle edgeH) const
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    return { m_mesh.face(heH), m_mesh.face(m_mesh.opposite_halfedge(heH)) };
}
template<typename BaseVecT>
void PmpMesh<BaseVecT>::getFacesOfVertex(VertexHandle handle, vector<FaceHandle>& facesOut) const
{
    for (const pmp::Face fH : m_mesh.faces(handle))
    {
        facesOut.push_back(fH);
    }
}
template<typename BaseVecT>
void PmpMesh<BaseVecT>::getEdgesOfVertex(VertexHandle handle, vector<EdgeHandle>& edgesOut) const
{
    for (const pmp::Halfedge heH : m_mesh.halfedges(handle))
    {
        edgesOut.push_back(m_mesh.edge(heH));
    }
}
template<typename BaseVecT>
void PmpMesh<BaseVecT>::getNeighboursOfVertex(VertexHandle handle, vector<VertexHandle>& verticesOut) const
{
    for (const pmp::Vertex vH : m_mesh.vertices(handle))
    {
        verticesOut.push_back(vH);
    }
}
template<typename BaseVecT>
OptionalFaceHandle PmpMesh<BaseVecT>::getOppositeFace(FaceHandle faceH, VertexHandle vertexH) const
{
    for (pmp::Halfedge heH : m_mesh.halfedges(faceH))
    {
        if (m_mesh.to_vertex(heH) == vertexH)
        {
            return m_mesh.face(m_mesh.opposite_halfedge(heH));
        }
    }
    return OptionalFaceHandle();
}
template<typename BaseVecT>
OptionalEdgeHandle PmpMesh<BaseVecT>::getOppositeEdge(FaceHandle faceH, VertexHandle vertexH) const
{
    for (pmp::Halfedge heH : m_mesh.halfedges(faceH))
    {
        if (m_mesh.to_vertex(heH) == vertexH)
        {
            return m_mesh.edge(m_mesh.opposite_halfedge(heH));
        }
    }
    return OptionalEdgeHandle();
}
template<typename BaseVecT>
OptionalVertexHandle PmpMesh<BaseVecT>::getOppositeVertex(FaceHandle faceH, EdgeHandle edgeH) const
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    if (m_mesh.face(heH) == faceH)
    {
        return m_mesh.to_vertex(heH);
    }
    heH = m_mesh.opposite_halfedge(heH);
    if (m_mesh.face(heH) == faceH)
    {
        return m_mesh.to_vertex(heH);
    }
    return OptionalVertexHandle();
}

template<typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> PmpMesh<BaseVecT>::verticesBegin() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<pmp::SurfaceMesh::VertexIterator>(m_mesh.vertices_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> PmpMesh<BaseVecT>::verticesEnd() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<pmp::SurfaceMesh::VertexIterator>(m_mesh.vertices_end())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> PmpMesh<BaseVecT>::facesBegin() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<pmp::SurfaceMesh::FaceIterator>(m_mesh.faces_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> PmpMesh<BaseVecT>::facesEnd() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<pmp::SurfaceMesh::FaceIterator>(m_mesh.faces_end())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> PmpMesh<BaseVecT>::edgesBegin() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<pmp::SurfaceMesh::EdgeIterator>(m_mesh.edges_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> PmpMesh<BaseVecT>::edgesEnd() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<pmp::SurfaceMesh::EdgeIterator>(m_mesh.edges_end())
    );
}

template<typename BaseVecT>
VertexSplitResult PmpMesh<BaseVecT>::splitVertex(VertexHandle vH)
{
    pmp::Halfedge longest_heH;
    double longest_length = -1;
    for (const pmp::Halfedge heH : m_mesh.halfedges(vH))
    {
        double length = m_mesh.edge_length(m_mesh.edge(heH));
        if (length > longest_length)
        {
            longest_heH = heH;
            longest_length = length;
        }
    }
    if (longest_length < 0)
    {
        panic("Called splitVertex on vertex with no edges");
    }

    EdgeSplitResult splitResult = this->splitEdge(m_mesh.edge(longest_heH));
    VertexSplitResult result(splitResult.edgeCenter);
    result.addedFaces.assign(splitResult.addedFaces.begin(), splitResult.addedFaces.end());

    // TODO: Check and fix Delaunay

    return result;
}

template<typename BaseVecT>
EdgeSplitResult PmpMesh<BaseVecT>::splitEdge(EdgeHandle eH)
{
    pmp::Point mid_point = (m_mesh.position(m_mesh.vertex(eH, 0)) + m_mesh.position(m_mesh.vertex(eH, 1))) / 2;
    pmp::Vertex mid_vH = m_mesh.add_vertex(mid_point);
    pmp::Halfedge new_heH = m_mesh.split(eH, mid_vH);

    EdgeSplitResult result(mid_vH);
    pmp::Face f = m_mesh.face(new_heH);
    if (f.is_valid())
    {
        result.addedFaces.push_back(f);
    }
    f = m_mesh.face(m_mesh.opposite_halfedge(new_heH));
    if (f.is_valid())
    {
        result.addedFaces.push_back(f);
    }
    return result;
}

template<typename BaseVecT>
void PmpMesh<BaseVecT>::fillHoles(size_t maxSize)
{
    DenseEdgeMap<bool> visitedEdges(m_mesh.edges_size(), false);
    std::vector<pmp::Halfedge> contours;

    for (const auto& eH : m_mesh.edges())
    {
        if (visitedEdges[eH])
        {
            continue;
        }
        visitedEdges[eH] = true;

        //get halfedges of edge
        auto heH = m_mesh.halfedge(eH, 0);
        if (!m_mesh.is_boundary(heH))
        {
            // if this HalfEdge has a face, check the other one
            heH = m_mesh.opposite_halfedge(heH);
            if (!m_mesh.is_boundary(heH))
            {
                // both sides have a face => not a boundary
                continue;
            }
        }

        // find contour vertices by running around the non-existing face (the hole) -> using .next
        auto start = heH;
        int count = 0;
        do
        {
            visitedEdges[m_mesh.edge(heH)] = true;
            heH = m_mesh.next_halfedge(heH);
            count++;
        } while (heH != start);

        // we only check maxSize after completing the above loop to ensure all edges are marked as visited
        if (count > maxSize || count < 3)
        {
            continue;
        }

        // as the contour fulfills all the necessary criteria, we add it to the list of contours which will be filled
        contours.push_back(heH);
    }

    cout << timestamp << "Found " << contours.size() << " holes" << endl;

    string comment = timestamp.getElapsedTime() + "Removing holes";
    ProgressBar progress(contours.size(), comment);

    pmp::SurfaceHoleFilling holeFilling(m_mesh);
    size_t filled = 0;

    // now fill the found holes
    for (pmp::Halfedge contour_heH : contours)
    {
        ++progress; // advance the progress bar
        try
        {
            holeFilling.fill_hole(contour_heH);
            filled++;
        }
        catch(pmp::InvalidInputException exception)
        {
            if (strcmp(exception.what(), "SurfaceHoleFilling: Non-manifold hole.") == 0)
            {
                // ignore non-manifold holes
                continue;
            }
            else
            {
                std::cerr << "Error while filling hole: " << exception.what() << std::endl;
            }
        }
    }
    cout << endl;
    cout << "Filled " << filled << " / " << contours.size() << " holes" << endl;
}

template<typename BaseVecT>
void PmpMesh<BaseVecT>::laplacianSmoothing(float smoothFactor, int numSmooths, bool useUniformLaplace)
{
    pmp::SurfaceSmoothing smoothing(m_mesh);
    smoothing.explicit_smoothing(numSmooths, smoothFactor, useUniformLaplace);
}

template<typename BaseVecT>
vector<VertexHandle> PmpMesh<BaseVecT>::findCommonNeigbours(VertexHandle vH1, VertexHandle vH2)
{
    std::unordered_set<pmp::Vertex> vH2nb;
    for (pmp::Vertex nb : m_mesh.vertices(vH2))
    {
        vH2nb.insert(nb);
    }
    vector<VertexHandle> result;
    for (pmp::Vertex nb : m_mesh.vertices(vH1))
    {
        if (vH2nb.find(nb) != vH2nb.end())
        {
            result.push_back(nb);
        }
    }
    return result;
}

template<typename BaseVecT>
void PmpMesh<BaseVecT>::splitVertex(EdgeHandle eH, VertexHandle vH, pmp::Point pos1, pmp::Point pos2)
{
    m_mesh.position(vH) = pos1;
    m_mesh.split(eH, pos2);
}

template<typename BaseVecT>
std::pair<BaseVecT, float> PmpMesh<BaseVecT>::triCircumCenter(FaceHandle faceH)
{
    //get vertices of the face
    auto v_iter = m_mesh.vertices(faceH);
    BaseVecT a = p2b(m_mesh.position(*v_iter));
    BaseVecT b = p2b(m_mesh.position(*(++v_iter)));
    BaseVecT c = p2b(m_mesh.position(*(++v_iter)));

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


} // namespace lvr2

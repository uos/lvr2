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
 * PMPMesh.tcc
 *
 *  @date 06.12.2021
 *  @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "lvr2/algorithm/pmp/SurfaceHoleFilling.h"
#include "lvr2/algorithm/pmp/SurfaceSmoothing.h"
#include "lvr2/algorithm/pmp/SurfaceSimplification.h"
#include "lvr2/util/Progress.hpp"

#include <unordered_set>

namespace lvr2
{

template<typename BaseVecT>
PMPMesh<BaseVecT>::PMPMesh(MeshBufferPtr ptr)
{
    size_t numFaces = ptr->numFaces();
    size_t numVertices = ptr->numVertices();

    auto vertices = ptr->getVertices();
    size_t i = 0;
    auto src_vertices = vertices.get();
    for (size_t i = 0; i < numVertices; i++, src_vertices += 3)
    {
        addVertex(BaseVecT(src_vertices[0], src_vertices[1], src_vertices[2]));
    }

    if (ptr->hasVertexNormals())
    {
        auto src_normals = ptr->getVertexNormals();
        auto dest_normals = m_mesh.vertex_property<pmp::Normal>("v:normal");
        auto src = src_normals.get();
        for (auto vH : m_mesh.vertices())
        {
            dest_normals[vH] = pmp::Normal(src[0], src[1], src[2]);
            src += 3;
        }
    }
    if (ptr->hasChannel<float>("texture_coordinates"))
    {
        auto src_texcoords = ptr->getTextureCoordinates();
        auto dest_texcoords = m_mesh.vertex_property<pmp::TexCoord>("v:tex");
        auto src = src_texcoords.get();
        for (auto vH : m_mesh.vertices())
        {
            dest_texcoords[vH] = pmp::TexCoord(src[0], src[1]);
            src += 2;
        }
    }
    if (ptr->hasVertexColors())
    {
        size_t w;
        auto src_colors = ptr->getVertexColors(w);
        if (w == 3 || w == 4)
        {
            auto dest_colors = m_mesh.vertex_property<pmp::Color>("v:color");
            auto src = src_colors.get();
            for (auto vH : m_mesh.vertices())
            {
                dest_colors[vH] = pmp::Color(src[0] / 255.0f, src[1] / 255.0f, src[2] / 255.0f);
                src += w;
            }
        }
    }

    auto indices = ptr->getFaceIndices();
    auto src_indices = indices.get();
    for (size_t i = 0; i < numFaces; i++, src_indices += 3)
    {
        VertexHandle v0(src_indices[0]);
        VertexHandle v1(src_indices[1]);
        VertexHandle v2(src_indices[2]);
        addFace(v0, v1, v2);
    }

    if (ptr->hasFaceNormals())
    {
        auto src_normals = ptr->getFaceNormals();
        auto dest_normals = m_mesh.face_property<pmp::Normal>("f:normal");
        auto src = src_normals.get();
        for (auto fH : m_mesh.faces())
        {
            dest_normals[fH] = pmp::Normal(src[0], src[1], src[2]);
            src += 3;
        }
    }
    if (ptr->hasFaceColors())
    {
        size_t w;
        auto src_colors = ptr->getFaceColors(w);
        if (w == 3 || w == 4)
        {
            auto dest_colors = m_mesh.face_property<pmp::Color>("f:color");
            auto src = src_colors.get();
            for (auto fH : m_mesh.faces())
            {
                dest_colors[fH] = pmp::Color(src[0] / 255.0f, src[1] / 255.0f, src[2] / 255.0f);
                src += w;
            }
        }
    }
}

template<typename BaseVecT>
EdgeCollapseResult PMPMesh<BaseVecT>::collapseEdge(EdgeHandle edgeH)
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
array<VertexHandle, 3> PMPMesh<BaseVecT>::getVerticesOfFace(FaceHandle handle) const
{
    auto f = m_mesh.vertices(handle);
    array<VertexHandle, 3> result = { *f, *(++f), *(++f) };
    return result;
}
template<typename BaseVecT>
array<EdgeHandle, 3> PMPMesh<BaseVecT>::getEdgesOfFace(FaceHandle handle) const
{
    auto f = m_mesh.halfedges(handle);
    array<EdgeHandle, 3> result = { m_mesh.edge(*f), m_mesh.edge(*(++f)), m_mesh.edge(*(++f)) };
    return result;
}
template<typename BaseVecT>
void PMPMesh<BaseVecT>::getNeighboursOfFace(FaceHandle handle, vector<FaceHandle>& facesOut) const
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
array<VertexHandle, 2> PMPMesh<BaseVecT>::getVerticesOfEdge(EdgeHandle edgeH) const
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    return { m_mesh.to_vertex(heH), m_mesh.from_vertex(heH) };
}
template<typename BaseVecT>
array<OptionalFaceHandle, 2> PMPMesh<BaseVecT>::getFacesOfEdge(EdgeHandle edgeH) const
{
    pmp::Halfedge heH = m_mesh.halfedge(edgeH, 0);
    return { m_mesh.face(heH), m_mesh.face(m_mesh.opposite_halfedge(heH)) };
}
template<typename BaseVecT>
void PMPMesh<BaseVecT>::getFacesOfVertex(VertexHandle handle, vector<FaceHandle>& facesOut) const
{
    for (const pmp::Face fH : m_mesh.faces(handle))
    {
        facesOut.push_back(fH);
    }
}
template<typename BaseVecT>
void PMPMesh<BaseVecT>::getEdgesOfVertex(VertexHandle handle, vector<EdgeHandle>& edgesOut) const
{
    for (const pmp::Halfedge heH : m_mesh.halfedges(handle))
    {
        edgesOut.push_back(m_mesh.edge(heH));
    }
}
template<typename BaseVecT>
void PMPMesh<BaseVecT>::getNeighboursOfVertex(VertexHandle handle, vector<VertexHandle>& verticesOut) const
{
    for (const pmp::Vertex vH : m_mesh.vertices(handle))
    {
        verticesOut.push_back(vH);
    }
}
template<typename BaseVecT>
OptionalFaceHandle PMPMesh<BaseVecT>::getOppositeFace(FaceHandle faceH, VertexHandle vertexH) const
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
OptionalEdgeHandle PMPMesh<BaseVecT>::getOppositeEdge(FaceHandle faceH, VertexHandle vertexH) const
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
OptionalVertexHandle PMPMesh<BaseVecT>::getOppositeVertex(FaceHandle faceH, EdgeHandle edgeH) const
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
MeshHandleIteratorPtr<VertexHandle> PMPMesh<BaseVecT>::verticesBegin() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<pmp::SurfaceMesh::VertexIterator>(m_mesh.vertices_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> PMPMesh<BaseVecT>::verticesEnd() const
{
    return MeshHandleIteratorPtr<VertexHandle>(
        std::make_unique<pmp::SurfaceMesh::VertexIterator>(m_mesh.vertices_end())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> PMPMesh<BaseVecT>::facesBegin() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<pmp::SurfaceMesh::FaceIterator>(m_mesh.faces_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> PMPMesh<BaseVecT>::facesEnd() const
{
    return MeshHandleIteratorPtr<FaceHandle>(
        std::make_unique<pmp::SurfaceMesh::FaceIterator>(m_mesh.faces_end())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> PMPMesh<BaseVecT>::edgesBegin() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<pmp::SurfaceMesh::EdgeIterator>(m_mesh.edges_begin())
    );
}
template<typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> PMPMesh<BaseVecT>::edgesEnd() const
{
    return MeshHandleIteratorPtr<EdgeHandle>(
        std::make_unique<pmp::SurfaceMesh::EdgeIterator>(m_mesh.edges_end())
    );
}

template<typename BaseVecT>
VertexSplitResult PMPMesh<BaseVecT>::splitVertex(VertexHandle split_vH)
{
    pmp::Halfedge longest_heH;
    double longest_length = -1;
    for (const pmp::Halfedge heH : m_mesh.halfedges(split_vH))
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

    // do an edge flip on the neighboring edges if the local delaunay criteria is not met
    vector<pmp::Vertex> commonVertexHandles = findCommonNeigbours(split_vH, m_mesh.to_vertex(longest_heH));

    EdgeSplitResult split_result = this->splitEdge(m_mesh.edge(longest_heH));
    VertexSplitResult result(split_result.edgeCenter);
    result.addedFaces = split_result.addedFaces;

    // check delaunay and flip edges if necessary
    for(pmp::Vertex vertex : commonVertexHandles)
    {
        OptionalEdgeHandle opt_eH = this->getEdgeBetween(vertex, split_vH);
        if (!opt_eH)
        {
            continue;
        }
        EdgeHandle eH = opt_eH.unwrap();
        if (!m_mesh.is_flip_ok(eH))
        {
            continue;
        }
        pmp::Halfedge heH = m_mesh.halfedge(eH, 0);
        pmp::Halfedge oH = m_mesh.opposite_halfedge(heH);

        //calculate the circumcenter of each triangle, look whether the local delaunay criteria is fulfilled
        auto circumCenterPair1 = triCircumCenter(m_mesh.face(heH));
        auto circumCenterPair2 = triCircumCenter(m_mesh.face(oH));

        BaseVecT circumCenter1 = circumCenterPair1.first;
        BaseVecT circumCenter2 = circumCenterPair2.first;

        float radius1 = circumCenterPair1.second;
        float radius2 = circumCenterPair2.second;

        BaseVecT opposite_vertex1 = getVertexPosition(m_mesh.to_vertex(m_mesh.next_halfedge(heH)));
        BaseVecT opposite_vertex2 = getVertexPosition(m_mesh.to_vertex(m_mesh.next_halfedge(oH)));

        // flip only, if one of the single vertices is inside the circumcircle of the other triangle
        if((opposite_vertex1-circumCenter2).length() <= radius2 || (opposite_vertex2-circumCenter1).length() <= radius1)
        {
            m_mesh.flip(eH);
        }
    }

    return result;
}

template<typename BaseVecT>
EdgeSplitResult PMPMesh<BaseVecT>::splitEdge(EdgeHandle eH)
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
void PMPMesh<BaseVecT>::fillHoles(size_t maxSize, bool simple)
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
    if (simple)
    {
        for (pmp::Halfedge contour_heH : contours)
        {
            ++progress;
            try
            {
                std::unordered_set<pmp::Vertex> seen;
                vector<pmp::Vertex> contour;
                pmp::Halfedge heH = contour_heH;
                do
                {
                    pmp::Vertex vH = m_mesh.to_vertex(heH);
                    if (seen.find(vH) != seen.end())
                    {
                        // broken hole: contains a loop
                        contour.clear();
                        break;
                    }
                    seen.insert(vH);
                    contour.push_back(vH);
                    heH = m_mesh.next_halfedge(heH);
                    assert(!m_mesh.face(heH).is_valid());
                } while (heH != contour_heH);

                if (contour.empty())
                {
                    continue;
                }

                //if the hole constist of three edges, we can instantly fill it by adding a face
                if (contour.size() == 3)
                {
                    addFace(contour[0], contour[1], contour[2]);
                    continue;
                }

                // calculate the averge point of the contour and adding it to the mesh
                pmp::Point middle = m_mesh.position(contour[0]);
                for (size_t i = 1; i < contour.size(); i++)
                {
                    middle += m_mesh.position(contour[i]);
                }
                middle /= contour.size();
                pmp::Vertex middle_vH = m_mesh.add_vertex(middle);

                pmp::Vertex prevH = contour.back();

                // add Triangles from adjacent vertices to the middle
                for (const auto& vH : contour)
                {
                    addFace(middle_vH, prevH, vH);
                    prevH = vH;
                }

                // apply a contour size dependent number of vertex splits to the mesh to add vertices to make the hole filling more smooth and consistent.
                for(int i = 0; i < contour.size(); i++)
                {
                    this->splitVertex(middle_vH);
                }
                filled++;
            }
            catch(PanicException exception)
            {
                std::cerr << "Error filling a hole: " << exception.what() << endl;
            }
            catch(pmp::TopologyException exception)
            {
                std::cerr << "Error filling a hole: " << exception.what() << endl;
            }
        }
    }
    else
    {
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
    }
    cout << endl;
    cout << "Filled " << filled << " / " << contours.size() << " holes" << endl;
}

template<typename BaseVecT>
void PMPMesh<BaseVecT>::laplacianSmoothing(float smoothFactor, int numSmooths, bool useUniformLaplace)
{
    pmp::SurfaceSmoothing smoothing(m_mesh);
    smoothing.explicit_smoothing(numSmooths, smoothFactor, useUniformLaplace);
}

template<typename BaseVecT>
vector<VertexHandle> PMPMesh<BaseVecT>::findCommonNeigbours(VertexHandle vH1, VertexHandle vH2)
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
void PMPMesh<BaseVecT>::splitVertex(EdgeHandle eH, VertexHandle vH, pmp::Point pos1, pmp::Point pos2)
{
    m_mesh.position(vH) = pos1;
    m_mesh.split(eH, pos2);
}

template<typename BaseVecT>
std::pair<BaseVecT, float> PMPMesh<BaseVecT>::triCircumCenter(FaceHandle faceH)
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

template<typename BaseVecT>
void PMPMesh<BaseVecT>::simplify(size_t targetNumVertices)
{
    pmp::SurfaceSimplification simplification(m_mesh);
    simplification.simplify(targetNumVertices);
}


} // namespace lvr2

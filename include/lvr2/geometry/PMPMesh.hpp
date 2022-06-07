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

/**
 * PMPMesh.hpp
 *
 * @date   06.12.2021
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#ifndef LVR2_GEOMETRY_PMPMESH_HPP_
#define LVR2_GEOMETRY_PMPMESH_HPP_

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/geometry/pmp/SurfaceMesh.h"

#include "lvr2/types/MeshBuffer.hpp"

namespace lvr2
{

/**
 * @brief Wrapper around the pmp::SurfaceMesh class to provide a lvr2::BaseMesh interface
 * 
 */
template<typename BaseVecT>
class PMPMesh : public BaseMesh<BaseVecT>
{
    static inline pmp::Point b2p(const BaseVecT& p)
    {
        return pmp::Point(p[0], p[1], p[2]);
    }
    static inline BaseVecT p2b(const pmp::Point& p)
    {
        return BaseVecT(p[0], p[1], p[2]);
    }
public:
    using Edge = pmp::Halfedge;
    using Face = pmp::Face;
    using Vertex = pmp::Vertex;

    PMPMesh()
    {}
    PMPMesh(MeshBufferPtr ptr);
    PMPMesh(const HighFive::Group& group);
    virtual ~PMPMesh() = default;

    /// Reads the mesh from a group. group must be written to with write(group)
    void read(const HighFive::Group& group);
    /// Writes the mesh into the given group
    void write(HighFive::Group& group) const;

    // ========================================================================
    // = Implementing the `BaseMesh` interface (see BaseMesh for docs)
    // ========================================================================

    VertexHandle addVertex(BaseVecT pos) override
    { return m_mesh.add_vertex(b2p(pos)); }
    FaceHandle addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H) override
    { return m_mesh.add_triangle(v1H, v2H, v3H); }
    void removeFace(FaceHandle handle) override
    { m_mesh.delete_face(handle); }
    EdgeCollapseResult collapseEdge(EdgeHandle edgeH) override;
    void flipEdge(EdgeHandle edgeH) override
    {
        if (!m_mesh.is_flip_ok(edgeH))
        {
            panic("flipEdge() called for non-flippable edge!");
        }
        m_mesh.flip(edgeH);
    }

    size_t numVertices() const override
    { return m_mesh.n_vertices(); }
    size_t numFaces() const override
    { return m_mesh.n_faces(); }
    size_t numEdges() const override
    { return m_mesh.n_edges(); }

    bool containsVertex(VertexHandle vH) const override
    {
        return m_mesh.is_valid(vH) && !m_mesh.is_deleted(vH);
    }
    bool containsFace(FaceHandle fH) const override
    {
        return m_mesh.is_valid(fH) && !m_mesh.is_deleted(fH);
    }
    bool containsEdge(EdgeHandle eH) const override
    {
        return m_mesh.is_valid(eH) && !m_mesh.is_deleted(eH);
    }

    bool isBorderEdge(EdgeHandle handle) const override
    { return m_mesh.is_boundary(handle); }
    bool isFlippable(EdgeHandle handle) const override
    { return m_mesh.is_flip_ok(handle); }

    Index nextVertexIndex() const override
    { return m_mesh.vertices_size(); }
    Index nextFaceIndex() const override
    { return m_mesh.faces_size(); }
    Index nextEdgeIndex() const override
    { return m_mesh.edges_size(); }

    BaseVecT getVertexPosition(VertexHandle handle) const override
    { return p2b(m_mesh.position(handle)); }
    BaseVecT& getVertexPosition(VertexHandle handle) override
    {
        // The returned reference might be modified by the caller, and those modifications must
        // be reflected in the original Point. Thus, we return a reference to the internal Point,
        // while pretending that it is a BaseVecT. This only works if the memory layout is the
        // same in both types, namely { <ScalarType> x, y, z } or equivalent.
        if constexpr (sizeof(BaseVecT) == sizeof(pmp::Point) && std::is_same_v<typename BaseVecT::CoordType, pmp::Point::value_type>)
        {
            return *(BaseVecT*)&m_mesh.position(handle);
        }
        else
        {
            panic("getVertexPosition() called for non-standard vertex type!");
        }
    }

    array<VertexHandle, 3> getVerticesOfFace(FaceHandle handle) const override;
    array<EdgeHandle, 3> getEdgesOfFace(FaceHandle handle) const override;
    void getNeighboursOfFace(FaceHandle handle, vector<FaceHandle>& facesOut) const override;
    array<VertexHandle, 2> getVerticesOfEdge(EdgeHandle edgeH) const override;
    array<OptionalFaceHandle, 2> getFacesOfEdge(EdgeHandle edgeH) const override;
    void getFacesOfVertex(VertexHandle handle, vector<FaceHandle>& facesOut) const override;
    void getEdgesOfVertex(VertexHandle handle, vector<EdgeHandle>& edgesOut) const override;
    void getNeighboursOfVertex(VertexHandle handle, vector<VertexHandle>& verticesOut) const override;
    OptionalFaceHandle getOppositeFace(FaceHandle faceH, VertexHandle vertexH) const override;
    OptionalEdgeHandle getOppositeEdge(FaceHandle faceH, VertexHandle vertexH) const override;
    OptionalVertexHandle getOppositeVertex(FaceHandle faceH, EdgeHandle edgeH) const override;

    MeshHandleIteratorPtr<VertexHandle> verticesBegin() const override;
    MeshHandleIteratorPtr<VertexHandle> verticesEnd() const override;
    MeshHandleIteratorPtr<FaceHandle> facesBegin() const override;
    MeshHandleIteratorPtr<FaceHandle> facesEnd() const override;
    MeshHandleIteratorPtr<EdgeHandle> edgesBegin() const override;
    MeshHandleIteratorPtr<EdgeHandle> edgesEnd() const override;


    // ========================================================================
    // = Other public methods
    // ========================================================================

    VertexSplitResult splitVertex(VertexHandle vertexToBeSplitH);
    EdgeSplitResult splitEdge(EdgeHandle edgeH);
    /**
     * @brief Fill holes smaller than maxSize
     * 
     * @param maxSize the maximum number of vertices around a hole. Bigger holes are ignored
     * @param simple true: simple but fast algorithm, false: more complex but slower algorithm
     */
    void fillHoles(size_t maxSize, bool simple = true);
    /**
     * @brief Performs Laplacian Smoothing
     * 
     * @param smoothFactor value between [0..1]. Used to determine the strength of each smoothing step. (usually 0.5)
     * @param numSmooths number of smoothing steps to perform.
     * @param useUniformLaplace true: use uniform weights (faster), false: use cotan weights (better quality)
     */
    void laplacianSmoothing(float smoothFactor = 0.5, int numSmooths = 1, bool useUniformLaplace = true);
    vector<VertexHandle> findCommonNeigbours(VertexHandle vH1, VertexHandle vH2);
    void splitVertex(EdgeHandle eH, VertexHandle vH, pmp::Point pos1, pmp::Point pos2);
    std::pair<BaseVecT, float> triCircumCenter(FaceHandle faceH);
    /**
     * @brief Decimates the Mesh with repeated Edge Collapses until the target number of vertices is reached
     * 
     * @param targetNumVertices the target number of vertices
     */
    void simplify(size_t targetNumVertices);

    pmp::SurfaceMesh& getSurfaceMesh()
    { return m_mesh; }
    const pmp::SurfaceMesh& getSurfaceMesh() const
    { return m_mesh; }

private:
    pmp::SurfaceMesh m_mesh;
};

} // namespace lvr2

#include "lvr2/geometry/PMPMesh.tcc"

#endif /* LVR2_GEOMETRY_PMPMESH_HPP_ */

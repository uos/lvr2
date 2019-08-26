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

//
// Created by Christian Swan on 09.10.17.
//

#include "lvr2/geometry/Handles.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/algorithm/ClusterAlgorithms.hpp"

// disable deprecation warnings for the Tesselator since GLUT is horrifying outdated.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <iostream>

namespace lvr2
{

template<typename BaseVecT>
GLUtesselator* Tesselator<BaseVecT>::m_tesselator;

template<typename BaseVecT>
std::vector<BaseVecT> Tesselator<BaseVecT>::m_vertices;

template<typename BaseVecT>
std::vector<BaseVecT> Tesselator<BaseVecT>::m_faces;

template<typename BaseVecT>
GLenum Tesselator<BaseVecT>::m_type;

template<typename BaseVecT>
void Tesselator<BaseVecT>::beginCallback(GLenum type)
{
    m_type = type;
    m_vertices.clear();
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::combineDataCallback(GLdouble coords[3],
                                               void *vertex_data[4],
                                               GLfloat weight[4],
                                               void **outData,
                                               void *userData)
{
    GLdouble* vertex = new GLdouble[3];

    vertex[0] = coords[0];
    vertex[1] = coords[1];
    vertex[2] = coords[2];

    BaseVecT v(vertex[0], vertex[1], vertex[2]);
    m_vertices.push_back(v);

    *outData = vertex;
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::endCallback(void)
{

    if (m_vertices.size() < 3)
    {
        m_vertices.clear();
        return;
    }

    if (m_type == GL_TRIANGLES)
    {
        for (size_t i = 0; i < m_vertices.size() / 3; ++i)
        {
            m_faces.push_back(m_vertices[i * 3 + 2]);
            m_faces.push_back(m_vertices[i * 3 + 1]);
            m_faces.push_back(m_vertices[i * 3 + 0]);
        }
    }
    else if (m_type == GL_TRIANGLE_FAN)
    {
        for(size_t i = 0; i < m_vertices.size() - 2; ++i)
        {
            m_faces.push_back(m_vertices[i + 2]);
            m_faces.push_back(m_vertices[i + 1]);
            m_faces.push_back(m_vertices[0]);
        }
    }
    else if (m_type == GL_TRIANGLE_STRIP)
    {
        for(size_t i = 0; i < m_vertices.size() - 2; ++i)
        {
            if (i % 2 == 0)
            {
                m_faces.push_back(m_vertices[i + 2]);
                m_faces.push_back(m_vertices[i + 1]);
                m_faces.push_back(m_vertices[i]);
            } else
            {
                m_faces.push_back(m_vertices[i]);
                m_faces.push_back(m_vertices[i + 1]);
                m_faces.push_back(m_vertices[i + 2]);
            }
        }
    }
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::errorCallback(GLenum errno)
{
    std::cerr << "[Tesselator-Error:] "
              << __FILE__ << " (" << __LINE__ << "): "
              << gluErrorString(errno) << std::endl;
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::vertexCallback(void* data)
{
    const GLdouble *ptr = (const GLdouble*)data;

    BaseVecT vertex(*ptr, *(ptr+1), *(ptr+2));
    m_vertices.push_back(vertex);
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::init() {
    // init tesselator
    if (m_tesselator)
    {
        gluDeleteTess(m_tesselator);
    }

    m_tesselator = gluNewTess();

    gluTessCallback(m_tesselator, GLU_TESS_BEGIN, (GLvoid (CALLBACK*) ()) &beginCallback);
    gluTessCallback(m_tesselator, GLU_TESS_VERTEX, (GLvoid (CALLBACK*) ()) &vertexCallback);
    gluTessCallback(m_tesselator, GLU_TESS_COMBINE_DATA, (GLvoid (CALLBACK*) ()) &combineDataCallback);
    gluTessCallback(m_tesselator, GLU_TESS_END, (GLvoid (CALLBACK*) ()) &endCallback);
    gluTessCallback(m_tesselator, GLU_TESS_ERROR, (GLvoid (CALLBACK*) ()) &errorCallback);

    gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::apply(
    BaseMesh<BaseVecT>& mesh,
    ClusterBiMap<FaceHandle>& clusters,
    DenseFaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals,
    float lineFusionThreshold
)
{
    // Status message for mesh generation
    string comment = timestamp.getElapsedTime() + "Tesselating clusters ";
    ProgressBar progress(clusters.numCluster(), comment);

    init();

    for (auto clusterH: clusters)
    {
        m_vertices.clear();
        m_faces.clear();

        auto contours = findContours(mesh, clusters, clusterH);
        // collect garbage GL vertices manually
        vector<GLdouble*> garbage;

        gluTessBeginPolygon(m_tesselator, 0);

        for (auto contour: contours)
        {
            if (contour.size() < 3)
            {
                continue;
            }

            // subtract lineFusionThreshold of lvr1 by one to avoid conflicts with new implementation
            auto simpleContour = simplifyContour(mesh, contour, 1 - lineFusionThreshold);

            gluTessBeginContour(m_tesselator);

            for(auto vH: simpleContour)
            {
                auto& v = mesh.getVertexPosition(vH);

                GLdouble* tVertex = new GLdouble[3];
                tVertex[0] = v.x;
                tVertex[1] = v.y;
                tVertex[2] = v.z;

                gluTessVertex(m_tesselator, tVertex, tVertex);
                garbage.push_back(tVertex);
            }

            gluTessEndContour(m_tesselator);
        }

        gluTessEndPolygon(m_tesselator);

        for (size_t i = 0; i < garbage.size(); i++)
        {
            delete[] garbage[i];
        }

        addTesselatedFaces(mesh, clusters, faceNormals, clusterH);

        ++progress;
    }

    gluDeleteTess(m_tesselator);
    m_tesselator = 0;

    if(!timestamp.isQuiet())
        cout << endl;
}

template<typename BaseVecT>
void Tesselator<BaseVecT>::addTesselatedFaces(
    BaseMesh<BaseVecT>& mesh,
    ClusterBiMap<FaceHandle>& clusters,
    DenseFaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals,
    ClusterHandle clusterH
)
{
    // delete all faces of cluster in mesh
    for (auto fH: clusters[clusterH].handles)
    {
        mesh.removeFace(fH);
    }

    // delete current cluster
    clusters.removeCluster(clusterH);

    // and than create new one
    auto newClusterH = clusters.createCluster();

    // keep track of the normal before to avoid non-normals on new faces...
    auto oldNormal = Normal<typename BaseVecT::CoordType>(0, 0, 1);

    // then re-add all faces and vertices generated by the tesselator
    for (size_t i = 0; i < m_faces.size() / 3; ++i)
    {
        auto v1 = m_faces[i * 3 + 2];
        auto v2 = m_faces[i * 3 + 1];
        auto v3 = m_faces[i * 3 + 0];

        // TODO make sure we reuse the added vertices here instead of duplicating everything
        auto v1H = mesh.addVertex(v1);
        auto v2H = mesh.addVertex(v2);
        auto v3H = mesh.addVertex(v3);

        if (!mesh.isFaceInsertionValid(v1H, v2H, v3H))
        {
            continue;
        }

        auto newFaceH = mesh.addFace(v1H, v2H, v3H);
        clusters.addToCluster(newClusterH, newFaceH);

        auto maybeNormal = getFaceNormal(mesh.getVertexPositionsOfFace(newFaceH));
        if (maybeNormal)
        {
            oldNormal = *maybeNormal;
        }

        faceNormals.insert(newFaceH, Normal<typename BaseVecT::CoordType>(maybeNormal ? *maybeNormal : oldNormal));
    }
}

} // namespace lvr2

#pragma GCC diagnostic pop

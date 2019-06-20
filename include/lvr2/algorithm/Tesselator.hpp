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

#ifndef LVR2_ALGORITHM_TESSELATOR_H
#define LVR2_ALGORITHM_TESSELATOR_H

#include "lvr2/geometry/Handles.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"

#if _MSC_VER
#include <Windows.h>
#endif

#ifndef __APPLE__
#include <GL/glu.h>
#include <GL/glut.h>
#define CALLBACK
#else
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#endif

namespace lvr2
{

/**
* Tesslation algorithm, utilizing the OpenGL tesselator to ease the reconstructed mesh.
* This algorithm is destryoing the mesh correlation between clusters, faces and vertices thus
* it is currently not suitable to run any algorithms requiring an coherent mesh.
*/
template<typename BaseVecT>
class Tesselator {

public:

    /**
     * Retesselates the current mesh cluster by cluster.
     */
    static void apply(
        BaseMesh<BaseVecT>& mesh,
        ClusterBiMap<FaceHandle>& clusters,
        DenseFaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals,
        float lineFusionThreshold
    );

private:

    /**
    * The OpenGL tesselation object
    */
    static GLUtesselator* m_tesselator;

    /**
    * The OpenGL type. This defines how the vertices are tesselated.
    */
    static GLenum m_type;

    /**
    * List of vertices to keep track of until the end of the tesselation.
    */
    static std::vector<BaseVecT> m_vertices;

    /**
    * List of faces to keep track of until the end of the tesselation.
    */
    static std::vector<BaseVecT> m_faces;

    /**
    * OpenGL callback, which is invoked by glBegin() to indicate the start of an primitiv.
    */
    static void CALLBACK beginCallback(GLenum type);

    /**
    * OpenGL callback, which is called for every vertex, adding it to our internal list.
    */
    static void CALLBACK vertexCallback(void* data);

    /**
    * OpenGL callback, which is called when the tesselator wants to create an new vertex on an intersection.
    * This simply takes the coords and adds it as an new vertex to our interal list.
    */
    static void CALLBACK combineDataCallback(GLdouble coords[3],
                                             void *vertex_data[4],
                                             GLfloat weight[4],
                                             void **outData,
                                             void *userData);

    /**
    * OpenGL callback, which is invoked by glEnd() to indicate the end of an primitiv.
    */
    static void CALLBACK endCallback(void);
    /**
    * OpenGL callback, which is called on any error occurring while running the tesslation.
    */
    static void CALLBACK errorCallback(GLenum errno);

    /**
    * Initalizes the tesselator object, deleting the old one if one is found and registering all
    * necessary callbacks.
    */
    static void init();

    /**
    * Adds the tesslated faces to the current cluster. Avoid any errors while adding
    * duplicated vertices, it first removes all of the faces in the cluster, than removing
    * the cluster itself and eventually creating an new one with the new tesslated faces.
    */
    static void addTesselatedFaces(
        BaseMesh<BaseVecT>& mesh,
        ClusterBiMap<FaceHandle>& clusters,
        DenseFaceMap<Normal<typename BaseVecT::CoordType>>& faceNormal,
        ClusterHandle clusterH
    );
};

} // namespace lvr2

#include "lvr2/algorithm/Tesselator.tcc"

#endif //LVR2_ALGORITHM_TESSELATOR_H

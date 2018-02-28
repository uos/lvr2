//
// Created by Christian Swan on 09.10.17.
//

#ifndef LVR2_ALGORITHM_TESSELATOR_H
#define LVR2_ALGORITHM_TESSELATOR_H

#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/ClusterBiMap.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>

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


template<typename BaseVecT>
class Tesselator {

public:

    /**
     * Retesselates the current mesh cluster by cluster.
     */
    static void apply(
        BaseMesh<BaseVecT>& mesh,
        ClusterBiMap<FaceHandle>& clusters,
        DenseFaceMap<Normal<BaseVecT>>& faceNormals,
        float lineFusionThreshold
    );

private:

    static GLUtesselator* m_tesselator;
    static GLenum m_type;
    static vector<Point<BaseVecT>> m_vertices;
    static vector<Point<BaseVecT>> m_faces;

    static void CALLBACK beginCallback(GLenum type);
    static void CALLBACK vertexCallback(void* data);

    static void CALLBACK combineDataCallback(GLdouble coords[3],
                                             void *vertex_data[4],
                                             GLfloat weight[4],
                                             void **outData,
                                             void *userData);
    static void CALLBACK endCallback(void);
    static void CALLBACK errorCallback(GLenum errno);

    static void init();
    static void addTesselatedFaces(
        BaseMesh<BaseVecT>& mesh,
        ClusterBiMap<FaceHandle>& clusters,
        DenseFaceMap<Normal<BaseVecT>>& faceNormal,
        ClusterHandle clusterH
    );
};

} // namespace lvr2

#include <lvr2/algorithm/Tesselator.tcc>

#endif //LVR2_ALGORITHM_TESSELATOR_H

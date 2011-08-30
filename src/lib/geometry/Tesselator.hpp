/*
 * Tesselator.hpp
 *
 * Created on: 28.08.2011
 *     Author: Florian Otte <fotte@uos.de>
 */

#ifndef TESSELATOR_H_
#define TESSELATOR_H_

using namespace std;

#include <vector>
#include <stack>
#include <glu.h>
#include <glut.h>

#include "Vertex.hpp"
#include "Normal.hpp"
//#include "BaseMesh.hpp"
#include "HalfEdgeVertex.hpp"
#include "HalfEdge.hpp"
//#include "HalfEdgeFace.hpp"
#include "Region.hpp"


namespace lssr
{

/* Forward Declarations */
template<typename VertexT, typename NormalT> class HalfEdgeVertex;


/**
 * @brief Tesselation class.
 *
 * Takes a list of vertices that describe the contour of a plane and
 * retesselates this contour to reduce the overall number of
 * triangles needed to represent this plane.
 */
template<typename VertexT, typename NormalT>
class Tesselator
{
public:
	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;


    /**
     * @brief Initializes the Tesselator
     */
    static void init(void);

    /**
     * @brief Takes a list of contours and retesselates the area.
     *
     * @param borderVertices A vector of stacks containing the contours.
     *                       The first stack is handled as the outer contour,
     *                       the rest are inner contours.
     *
     * @return Returns a list of HalfEdgeVertices. Every 3-points represent a triangle.
     *         
     */
    //static vector<HVertex> tesselate(vector<stack<HVertex*> > borderVertices);
    static void tesselate(vector<stack<HVertex*> > borderVertices);
    
    /**
     * @brief Takes a list of contours and retesselates the area.
     *
     * @param region An object of the Region class. 
     *               This represents the region which should be retesselated
     *
     * @return Returns a list of HalfEdgeVertices. Every 3-points represent a triangle.
     *         
     */
    //static vector<HVertex> tesselate(Region<VertexT, NormalT> region);
    static void tesselate(Region<VertexT, NormalT> region);

    /**
     * @brief blabla
     *
     * @param vertices a double array containing all vertices.
     * @param vLength  the length of the vertice array. /3 == numberVertices
     * @param faces    list of faces. this is just an indexlist pointing to the vertices array.
     *                 for face i: vertices[i+0] == first vertex for this face.
     *                             vertices[i+1] == second ....
     *                             vertices[i+2] == third...
     *
     *
     */
    static void getFinalizedTriangles(double **vertexBuffer,
                                      double **normalBuffer,
                                      double **colorBuffer,
                                      uint8_t   **indexBuffer,
                                      int *numberFaces,
                                      int *numVertices);
    
    /**
     * @Brief Callback function
     *
     * @param which The type of Triangles that result from this polygon.
     *              This maybe simple triangles, a triangle-fan, or a triangle-stripe.
     *
     * @param userData A pointer to user defined Data. These userData are result from the use
     *                 of gluTessVertex(tessObject, pointData, userData);
     */
    static void tesselatorBegin(GLenum which, HVertex* userData);

    /**
     * @Brief Callback function
     */
    static void tesselatorEnd();

    /**
     * @Brief Callback function
     *
     * @param errorCode The error that arose during tesselation. Thrown by the a tesselation object.
     */
    static void tesselatorError(GLenum errorCode);

    /**
     * @Brief Callback function
     */
    static void tesselatorAddVertex(const GLvoid *data, HVertex* userData);

    /**
     * @Brief Callback function
     */
    static void tesselatorCombineVertices(GLdouble coords[3],
							 GLdouble *vertex_data[4],
							 GLfloat weight[4],
							 GLdouble **dataOut,
                             HVertex* userData);

    static bool   m_tesselated;
    static int    m_region;
    //static vector<Vertex<float> > m_vertices;
    static vector<HVertex> m_vertices;
    static vector<Vertex<float> > m_triangles;
    static GLUtesselator* m_tesselator;

    static GLenum m_primitive;
    static int    m_numContours;    
    static bool   m_debug;
    static NormalT m_normal;

private:
    /* All Constructors shall be private since
       this class is just a collection of functions. */
    Tesselator();
    ~Tesselator();
    Tesselator(const Tesselator& rhs);

}; /* end of class */

} /* namespace lssr */

#include "Tesselator.tcc"
#endif /* TESSELATOR_H_ */

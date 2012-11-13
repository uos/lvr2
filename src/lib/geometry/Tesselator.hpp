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
 * Tesselator.hpp
 *
 *  @date 28.08.2011
 *  @author Florian Otte <fotte@uos.de>
 */

#ifndef TESSELATOR_H_
#define TESSELATOR_H_

using namespace std;

#include <vector>
#include <glu.h>
#include <glut.h>
#include <iomanip>

#include "Vertex.hpp"
#include "Normal.hpp"
#include "HalfEdge.hpp"
#include "Region.hpp"

namespace lssr
{



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

    /**
     * @brief Initializes the Tesselator
     *
     * This is necessary before using the tesselator functions.
     */
    static void init(void);

    /**
     * @brief Takes a list of contours and retesselates the area.
     *
     * @param borderVertices A vector of vectors containing the contours.
     *                       The first stack is handled as the outer contour,
     *                       the rest are inner contours.
     *
     * @return Returns a list of Vertices. Every 3-points represent a triangle.
     *         
     */
    static void tesselate(vector<vector<VertexT> > &borderVertices);
    
    /**
     * @brief Takes a list of contours and retesselates the area.
     *
     * @param region An object of the Region class. 
     *               This represents the region which should be retesselated
     *         
     */
    static void tesselate(Region<VertexT, NormalT> *region);

    /**
     * @brief blabla
     *
     * @param vertices a float array containing all vertices.
     * @param vLength  the length of the vertice array. /3 == numberVertices
     * @param faces    list of faces. this is just an indexlist pointing to the vertices array.
     *                 for face i: vertices[i+0] == first vertex for this face.
     *                             vertices[i+1] == second ....
     *                             vertices[i+2] == third...
     *
     *
     */
    static void getFinalizedTriangles(vector<float> &vertexBuffer, vector<unsigned int> &indexBuffer, vector<vector<VertexT> > &vectorBorderPoints);


private:
    
    /**
     * @Brief Callback function
     *
     * @param which The type of Triangles that result from this polygon.
     *              This maybe simple triangles, a triangle-fan, or a triangle-stripe.
     *
     * @param userData A pointer to user defined Data. These userData are result from the use
     *                 of gluTessVertex(tessObject, pointData, userData);
     */
    static void tesselatorBegin(GLenum which, VertexT* userData);

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
    static void tesselatorAddVertex(const GLvoid *data, VertexT* userData);

    /**
     * @Brief Callback function
     */
    static void tesselatorCombineVertices(GLdouble coords[3],
							 GLdouble *vertex_data[4],
							 GLdouble weight[4],
							 GLdouble **dataOut,
                             VertexT* userData);
    /* All Constructors shall be private since
       this class is just a collection of functions. */
    Tesselator();

    ~Tesselator();

    Tesselator(const Tesselator& rhs);


    /** Variable declarations */

    /* The tesselation object */
    static GLUtesselator* m_tesselator;

    /* List of vertices. used to keep track until tesselation ends */
    static vector<VertexT> m_vertices;

    /* List of triangles. used to keep track of triangles until tesselation ends */
    static vector<Vertex<float> > m_triangles;

    /* The current primitive-type. */
    static GLenum m_primitive;

    /* Number of already tesselated contours. Used for debugging */
    static int    m_numContours;    
}; /* end of class */

} /* namespace lssr */

#include "Tesselator.tcc"
#endif /* TESSELATOR_H_ */

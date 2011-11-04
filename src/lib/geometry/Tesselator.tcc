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
 * Tesselator.tcc
 *
 * Created on: 29.08.2011
 *     Author: Florian Otte <fotte@uos.de>
 */

#ifndef TESSELATOR_C_
#define TESSELATOR_C_

using namespace std;

namespace lssr
{

/* Deklaration of all static data */
template<typename VertexT, typename NormalT> vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::m_vertices;
template<typename VertexT, typename NormalT> vector<Vertex<float> > Tesselator<VertexT, NormalT>::m_triangles;
template<typename VertexT, typename NormalT> GLUtesselator* Tesselator<VertexT, NormalT>::m_tesselator;
template<typename VertexT, typename NormalT> GLenum  Tesselator<VertexT, NormalT>::m_primitive;
template<typename VertexT, typename NormalT> int     Tesselator<VertexT, NormalT>::m_numContours=0;


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorBegin(GLenum which, HVertex* userData)
{
    m_primitive = which;
    m_vertices.clear();
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorEnd()
{
    if(m_vertices.size() < 3)
    {
        cerr << "Only 3 Points after retriangulation. Aborting.\n";
        m_vertices.clear();
        return;
    }

#ifdef DB_TESS
    stringstream tFileName; tFileName << "tesselationResult_" << setw(4) << setfill('0') << m_numContours << ".txt";
    ofstream tess(tFileName.str().c_str(), ios_base::app);
#endif

    if(m_primitive == GL_TRIANGLES)
    {
        for(size_t i=0; i<m_vertices.size() / 3; ++i)
        {
            m_triangles.push_back((m_vertices[i*3+2]).m_position);
            m_triangles.push_back((m_vertices[i*3+1]).m_position);
            m_triangles.push_back((m_vertices[i*3+0]).m_position);

#ifdef DB_TESS
            tess << m_vertices[i].m_position[0] << " " << m_vertices[i].m_position[1] << " " << m_vertices[i].m_position[2] << endl;
            if((i+1)%3==0 && i != 0)
                tess << m_vertices[i-2].m_position[0] << " " << m_vertices[i-2].m_position[1] << " " << m_vertices[i-2].m_position[2] << "\n#EndTriangle<<\n\n";
#endif
        } 

    } else if(m_primitive == GL_TRIANGLE_FAN)
    {
        for(size_t i=0; i<m_vertices.size()-2; ++i)
        {
            m_triangles.push_back((m_vertices[i+2]).m_position);
            m_triangles.push_back((m_vertices[i+1]).m_position);
            m_triangles.push_back((m_vertices[0]).m_position);

#ifdef DB_TESS
            tess << m_vertices[0].m_position[0]   << " " << m_vertices[0].m_position[1]   << " " << m_vertices[0].m_position[2]   << endl
                << m_vertices[i+1].m_position[0] << " " << m_vertices[i+1].m_position[1] << " " << m_vertices[i+1].m_position[2] << endl
                << m_vertices[i+2].m_position[0] << " " << m_vertices[i+2].m_position[1] << " " << m_vertices[i+2].m_position[2] << endl
                << m_vertices[0].m_position[0]   << " " << m_vertices[0].m_position[1]   << " " << m_vertices[0].m_position[2] << "\n#EndTriangle<<\n\n";
#endif
        } 
    } else if(m_primitive == GL_TRIANGLE_STRIP)
    {
        for(size_t i=0; i<m_vertices.size()-2; ++i)
        {
            if(i%2 ==  0)
            {
                m_triangles.push_back((m_vertices[i+2]).m_position);
                m_triangles.push_back((m_vertices[i+1]).m_position);
                m_triangles.push_back((m_vertices[i]).m_position);
            } else
            {
                m_triangles.push_back((m_vertices[i]).m_position);
                m_triangles.push_back((m_vertices[i+1]).m_position);
                m_triangles.push_back((m_vertices[i+2]).m_position);
            }

#ifdef DB_TESS
            tess << m_vertices[i].m_position[0]   << " " << m_vertices[i].m_position[1]   << " " << m_vertices[i].m_position[2]   << endl
                << m_vertices[i+1].m_position[0] << " " << m_vertices[i+1].m_position[1] << " " << m_vertices[i+1].m_position[2] << endl
                << m_vertices[i+2].m_position[0] << " " << m_vertices[i+2].m_position[1] << " " << m_vertices[i+2].m_position[2] << endl
                << m_vertices[i].m_position[0]   << " " << m_vertices[i].m_position[1]   << " " << m_vertices[i].m_position[2]   << "\n#EndTriangle<<\n\n";
#endif
        }
    }

#ifdef DB_TESS
    tess << "#EndPoly<<" << endl;
    tess.close();
#endif
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorError(GLenum errorCode)
{
    cerr << "[Tesselator-Error:] " << __FILE__ << " (" << __LINE__ << "): " << gluErrorString(errorCode) << endl; 
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorAddVertex(const GLvoid *data, HVertex* userData)
{
    const GLdouble *ptr = (const GLdouble*)data;
    Vertex<float> v(*ptr, *(ptr+1), *(ptr+2));
    HVertex newVertex(v);
    m_vertices.push_back(newVertex);
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::getFinalizedTriangles(vector<float> &vertexBuffer, vector<unsigned int> &indexBuffer, vector<vector<HVertex*> > &vectorBorderPoints)
{
    // used typedefs
    typedef vector<Vertex<float> >::iterator vertexIter;
    // make sure everything is initialized
    init();

    // retesselate the contours
    tesselate(vectorBorderPoints);

    // keep track of already used vertices to avoid doubled or tripled vertices
    vector<Vertex<float> > usedVertices;
    size_t search;

    // iterate over all new triangles:
    vertexIter triangles=m_triangles.begin(), trianglesEnd=m_triangles.end();

    int posArr[3]; posArr[0]=-1; posArr[1]=-1; posArr[2]=-1;
    // add all triangles and so faces to our buffers and keep track of all used parameters
    int m=0;

    for(; triangles != trianglesEnd; ++triangles)
    {
        //search = ( std::find(usedVertices.begin(), usedVertices.end(), (*triangles) ) - usedVertices.begin() );
        //if(search!=usedVertices.size())
        //{
        //    posArr[m] = search;
        //    cout << "Found: " << " " << *triangles << endl << "\t Search: " << search << endl;

        //   vector<Vertex<float> >::iterator mm=usedVertices.begin();
        //    for(; mm != usedVertices.end(); ++mm)
        //        cout <<"\t"<< *mm;
        //} else
        { 
            vertexBuffer.push_back((*triangles)[0]);
            vertexBuffer.push_back((*triangles)[1]);
            vertexBuffer.push_back((*triangles)[2]);
            posArr[m] = usedVertices.size();
            usedVertices.push_back(*triangles);
        }

        m++;

        if(m == 3) // we added 3 vertices therefore a whole face!!
        {
            indexBuffer.push_back(posArr[0]);
            indexBuffer.push_back(posArr[1]);
            indexBuffer.push_back(posArr[2]);

            m=0;
        }
    }
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorCombineVertices(GLdouble coords[3],
        GLdouble *vertex_data[4],
        GLdouble weight[4],
        GLdouble **dataOut,
        HVertex* userData)
{

    GLdouble *vertex = (GLdouble*) malloc(6*sizeof(GLdouble));

    if(!vertex)
    {
        cerr << "Could not allocate memory - undefined behaviour will/might arise from now on!" << endl;
    }
    vertex[0] = coords[0];
    vertex[1] = coords[1];
    vertex[2] = coords[2];

    Vertex<float> v(coords[0], coords[1], coords[2]);
    m_vertices.push_back(HVertex(v));
    *dataOut = vertex;
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::init(void)
{
    m_vertices.clear();
    m_triangles.clear();
    if( m_tesselator )
    {
        return;
    }

    if( GLU_VERSION < 1.1)
    {
        cerr<< "Unsupported Version of GLUT." 
            << "Please use OpenGL Utility Library version 1.1 or higher." << endl;
        return;
    }

    m_tesselator = gluNewTess();
    if(!m_tesselator)
    {
        cerr<<"Could not allocate tesselation object. Aborting tesselation." << endl;
        return;
    }

    /* Callback function that define beginning of polygone etc. */
    gluTessCallback(m_tesselator, GLU_TESS_VERTEX_DATA,(GLvoid(*) ()) &tesselatorAddVertex);
    gluTessCallback(m_tesselator, GLU_TESS_BEGIN_DATA, (GLvoid(*) ()) &tesselatorBegin);
    gluTessCallback(m_tesselator, GLU_TESS_END, (GLvoid(*) ()) &tesselatorEnd);
    gluTessCallback(m_tesselator, GLU_TESS_COMBINE_DATA, (GLvoid(*) ()) &tesselatorCombineVertices);
    gluTessCallback(m_tesselator, GLU_TESS_ERROR, (GLvoid(*) ()) &tesselatorError);


    /* set Properties for tesselation */
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_POSITIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NEGATIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
    gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD);
    /* Use gluTessNormal: speeds up the tessellation if the
       Polygon lies on a x-y plane. and it approximatly does!*/
    //gluTessNormal(m_tesselator, 0, 0, 1);
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselate(vector<vector<HVertex*> > vectorBorderPoints)
{
    if(!m_tesselator)
    {
        cerr<<"No Tesselation Object Created. Please use Tesselator::init() before making use of Tesselator::tesselate(...). Aborting Tesselation." << endl;
        return; 
    }

    if(!vectorBorderPoints.size())
    {
        cerr<< "No points received. Aborting Tesselation." << endl;
        return;
    } 

#ifdef DB_TESS
    cout << "Received " << vectorBorderPoints.size() << " contours to retesselate.\n";
#endif



    /* Begin definition of the polygon to be tesselated */
    gluTessBeginPolygon(m_tesselator, 0);

    for(size_t i=0; i<vectorBorderPoints.size(); ++i)
    {
        vector<HVertex*> borderPoints = vectorBorderPoints[i];
#ifdef DB_TESS
        stringstream tFileName; tFileName << "contour[" << 
            setw(3) << setfill ('0') << i << "]_size["<< borderPoints.size() << "].txt";
        ofstream orgContour(tFileName.str().c_str());
#endif

        if(borderPoints.size() <3 )
        {
#ifdef DB_TESS
            cout << "BorderContains less than 3 Points!. Aborting.\n";
            cout << "i: " << i << " size:" << borderPoints.size() << endl;
            orgContour.close();
#endif
            continue; 
        }

///        HVertex* contourBegin = borderPoints.back();

        // Begin Contour
        gluTessBeginContour(m_tesselator);

        while(borderPoints.size() > 0)
        {
#ifdef DB_TESS
            orgContour << (borderPoints.back())->m_position[0] << " " <<  (borderPoints.back())->m_position[1] << " " << (borderPoints.back())->m_position[2] << endl;
#endif

            GLdouble* vertex = new GLdouble[3];
            vertex[0] = (borderPoints.back())->m_position[0];
            vertex[1] = (borderPoints.back())->m_position[1];
            vertex[2] = (borderPoints.back())->m_position[2];
            borderPoints.pop_back();
            gluTessVertex(m_tesselator, vertex, vertex);
        }

        /* End Contour */
        gluTessEndContour(m_tesselator);
#ifdef DB_TESS
        orgContour << contourBegin->m_position[0] << " " << contourBegin->m_position[1] << " " << contourBegin->m_position[2];
        orgContour.close();
#endif
    }

    /* End Tesselation */
    gluTessEndPolygon(m_tesselator);
    m_numContours++;
    gluDeleteTess(m_tesselator);
    m_tesselator = 0;
    return;
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselate(Region<VertexT, NormalT> *region)
{
    tesselate(region->getContours(0.01));
}

} /* namespace lssr */
#endif /* TESSELATOR_C */

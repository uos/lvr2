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

//template<typename VertexT, typename NormalT> typedef HalfEdgeVertex<VertexT, NormalT> HVertex;
template<typename VertexT, typename NormalT> vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::m_vertices;
template<typename VertexT, typename NormalT> vector<Vertex<float> > Tesselator<VertexT, NormalT>::m_triangles;
template<typename VertexT, typename NormalT> GLUtesselator* Tesselator<VertexT, NormalT>::m_tesselator;
template<typename VertexT, typename NormalT> GLenum  Tesselator<VertexT, NormalT>::m_primitive;
template<typename VertexT, typename NormalT> NormalT Tesselator<VertexT, NormalT>::m_normal;
template<typename VertexT, typename NormalT> int     Tesselator<VertexT, NormalT>::m_numContours;
template<typename VertexT, typename NormalT> bool    Tesselator<VertexT, NormalT>::m_debug; 
template<typename VertexT, typename NormalT> bool    Tesselator<VertexT, NormalT>::m_tesselated;
template<typename VertexT, typename NormalT> int     Tesselator<VertexT, NormalT>::m_region;

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorBegin(GLenum which, HVertex* userData)
{
    m_primitive = which;
    m_vertices.clear();
}

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorEnd()
{
    stringstream tFileName; tFileName << "tess_" << m_numContours << ".txt";
    ofstream tess(tFileName.str().c_str(), ios_base::app);
    
    if(m_primitive == GL_TRIANGLES)
    {
        for(int i=0; i<m_vertices.size(); ++i)
        {
            m_triangles.push_back((m_vertices[i]).m_position);
/*            if(m_debug)
            {
                tess << m_vertices[i][0] << " " << m_vertices[i][1] << " " << m_vertices[i][2] << endl;
                if((i+1)%3==0 && i != 0)
                    tess << m_vertices[i-2][0] << " " << m_vertices[i-2][1] << " " << m_vertices[i-2][2] << "\n#EndTriangle<<\n\n";
            } */
        }
    } else if(m_primitive == GL_TRIANGLE_FAN)
    {
        for(int i=0; i<m_vertices.size()-2; ++i)
        {
            m_triangles.push_back((m_vertices[0]).m_position);
            m_triangles.push_back((m_vertices[i+1]).m_position);
            m_triangles.push_back((m_vertices[i+2]).m_position);
/*            if(m_debug)
            {
            tess << m_vertices[0][0]   << " " << m_vertices[0][1]   << " " << m_vertices[0][2]   << endl
                 << m_vertices[i+1][0] << " " << m_vertices[i+1][1] << " " << m_vertices[i+1][2] << endl
                 << m_vertices[i+2][0] << " " << m_vertices[i+2][1] << " " << m_vertices[i+2][2] << endl
                 << m_vertices[0][0]   << " " << m_vertices[0][1]   << " " << m_vertices[0][2] << "\n#EndTriangle<<\n\n";
            } */
        }
    } else if(m_primitive == GL_TRIANGLE_STRIP)
    {
        for(int i=0; i<m_vertices.size()-2; ++i)
        {
            m_triangles.push_back((m_vertices[i]).m_position);
            m_triangles.push_back((m_vertices[i+1]).m_position);
            m_triangles.push_back((m_vertices[i+2]).m_position);
/*            if(m_debug)
            {
            tess << m_vertices[i][0]   << " " << m_vertices[i][1]   << " " << m_vertices[i][2]   << endl
                 << m_vertices[i+1][0] << " " << m_vertices[i+1][1] << " " << m_vertices[i+1][2] << endl
                 << m_vertices[i+2][0] << " " << m_vertices[i+2][1] << " " << m_vertices[i+2][2] << endl
                 << m_vertices[i][0]   << " " << m_vertices[i][1]   << " " << m_vertices[i][2]   << "\n#EndTriangle<<\n\n";
            } */
        }
    }
    if(m_debug)
    {
        tess << "#EndPoly<<" << endl;
    }
	tess.close();
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
    //Vertex<double> v(*ptr, *(ptr+1), *(ptr+2));
    Vertex<float> v(*ptr, *(ptr+1), *(ptr+2));
    HVertex newVertex(v);
    newVertex.m_normal = userData->m_normal;
    m_vertices.push_back(newVertex);
}

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::getFinalizedTriangles(double **vertexBuffer,
                                      double **normalBuffer,
                                      double **colorBuffer,
                                      uint8_t   **indexBuffer,
                                      int *lengthFaces,
                                      int *lengthVertices)
{
    // make a good guess how long the normal buffer is supposed to be
    // colorbuffer, vertexbuffer a.s.o.
    // Vertexbuffer: 3 entries are 1 vertex -> [X1, Y1, Z1, X2, Y2, Z2, X3....]
    // normalbuffer: 3 entries are 1 Normal -> [n1, n1, n1, n2, n2, n2, ......]
    // colorbuffer:  3 etnries are 1 vertex -> [r1, g1, b1, ..................]
    //
    // indexbuffer: every vertice has an index in vertexbuffer. 
    //
    // good guess: #faces * 3 == #vertices
    // good guess: len(VertexBuffer) = #vertices * 3
    uint16_t numFaces = m_triangles.size() / 3;
    uint16_t numVertices = m_triangles.size();

    // allocate new memory.
    *vertexBuffer = new double[numVertices*3];
    *normalBuffer = new double[numVertices*3];
    *colorBuffer  = new double[numVertices*3];

    *indexBuffer =  new uint8_t[numVertices];

    uint16_t usedVertices=0, usedNormals=0, usedColors=0, usedFaces=0;
    
    // keep track of already used vertices to avoid doubles.
    vector<Vertex<double> > vertices;
    vector<Vertex<double> >::iterator triangles    = m_triangles.begin();
    vector<Vertex<double> >::iterator trianglesEnd = m_triangles.end();

    int count=0;
    int posArr[3]; posArr[0]=-1; posArr[1]=-1; posArr[2]=-1;
    // add all triangles and so faces to our buffers and keep track of all used parameters
    for(; triangles != trianglesEnd; ++triangles)
    {
        // try to find the new triangleVertex in the list of used vertices.
        vector<Vertex<double> >::iterator it    = vertices.begin();
        vector<Vertex<double> >::iterator itEnd = vertices.end();
        int pos=0;
        while(it != itEnd && *it != *triangles) 
        {
            it++;
            pos++;
        }
        if(it != itEnd)
        {
            posArr[count%3] = pos;
        } else
        {
            // vertex was not used before so store it;
            vertices.push_back(*triangles);
            vertexBuffer[(usedVertices * 3) + 0] = (*triangles)[0]; 
            vertexBuffer[(usedVertices * 3) + 1] = (*triangles)[1];
            vertexBuffer[(usedVertices * 3) + 2] = (*triangles)[2];

            normalBuffer[(usedVertices *3) + 0] = m_normal[0];
            normalBuffer[(usedVertices *3) + 1] = m_normal[1];
            normalBuffer[(usedVertices *3) + 2] = m_normal[2];

            posArr[count%3] = usedVertices;
            usedVertices++;
        }
        
        if((count+1)%3 == 0) // we added 3 vertices therefore a whole face!!
        {
            indexBuffer[(usedFaces * 3) + 0] = posArr[0]; 
            indexBuffer[(usedFaces * 3) + 1] = posArr[1];
            indexBuffer[(usedFaces * 3) + 2] = posArr[2];
            posArr[0]=-1; posArr[1]=-1; posArr[2]=-1;

            int r,g,b, surface_class = m_region;
            if(surface_class != INT_MAX)
            {
                r = fabs(cos(surface_class)); 
                g = fabs(sin(surface_class * 30));
                b = fabs(sin(surface_class * 2));
            } else
            {
                r = 0.0;
                g = 0.8;
                b = 0.0;
            }
            colorBuffer[posArr[0] * 3 + 0] = r;
            colorBuffer[posArr[0] * 3 + 1] = g;
            colorBuffer[posArr[0] * 3 + 2] = b;

            colorBuffer[posArr[1] * 3 + 0] = r;
            colorBuffer[posArr[1] * 3 + 1] = g;
            colorBuffer[posArr[1] * 3 + 2] = b;

            colorBuffer[posArr[2] * 3 + 0] = r;
            colorBuffer[posArr[2] * 3 + 1] = g;
            colorBuffer[posArr[2] * 3 + 2] = b;
            
            usedFaces++;
        }
        count++;
    }

    // Copy all that stuff and resize array -- this should be improved somehow! TODO:!
    double *newVertexBuffer = new double[usedVertices*3];
    double *newNormalBuffer = new double[usedVertices*3];
    double *newColorBuffer  = new double[usedVertices*3];
    uint8_t *newIndexBuffer = new uint8_t[usedFaces*3];

    // use memcopy?
    for(int i=0; i<usedVertices*3; i++)
    {
        newVertexBuffer[i] = vertexBuffer[i];
        newNormalBuffer[i] = normalBuffer[i];
        newColorBuffer[i]  = colorBuffer[i];
    }

    for(int i=0; i<usedFaces*3; ++i)
    {
        newIndexBuffer[i] = indexBuffer[i];
    }
    
    delete *indexBuffer;
    delete *vertexBuffer;
    delete *normalBuffer;
    delete *colorBuffer;
    *vertexBuffer = newVertexBuffer;
    *normalBuffer = newNormalBuffer;
    *colorBuffer  = newColorBuffer;
    *indexBuffer = newIndexBuffer;
    *lengthVertices = usedVertices*3;
    *lengthFaces = usedFaces*3;

    if(m_debug)
    {
        cout << "Retesselation Complete. " << usedVertices << " Vertices. " << usedFaces << " Faces.\n";
    }
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorCombineVertices(GLdouble coords[3],
							 GLdouble *vertex_data[4],
							 GLfloat weight[4],
							 GLdouble **dataOut,
                             HVertex* userData)
{
    
	GLdouble *vertex = (GLdouble*) malloc(6*sizeof(GLdouble));
    GLdouble *ptr = coords;
	
	if(!vertex)
	{
		cerr << "Could not allocate memory - undefined behaviour will/might arise from now on!" << endl;
	}
	vertex[0] = coords[0];
	vertex[1] = coords[1];
	vertex[2] = coords[2];
    
    Vertex<float> v(coords[0], coords[1], coords[2]);
    HVertex newVertex(v);
    newVertex.m_normal = userData->m_normal;

    m_vertices.push_back(newVertex);
	*dataOut = vertex;
}

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::init(void)
{
    m_vertices.clear();
    m_triangles.clear();
    m_numContours=0;
    m_tesselated = false;
    m_region = INT_MAX;
    
    // NOTE: Replace the 2 by the correct glueGetString.
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
    gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_POSITIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NEGATIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD);
	 /* Use gluTessNormal: speeds up the tessellation if the
	  	Polygon lies on a x-y plane. and it approximatly does!*/
	 //gluTessNormal(tesselator, 0, 0, 1);
}

template<typename VertexT, typename NormalT>
//vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::tesselate(vector<stack<HVertex*> > vectorBorderPoints)
void Tesselator<VertexT, NormalT>::tesselate(vector<stack<HVertex*> > vectorBorderPoints)
{
    if(!m_tesselator)
    {
        cerr<<"No Tesselation Object Created. Please use Tesselator::init() before making use of Tesselator::tesselate(...). Aborting Tesselation." << endl;
        return ; //vector<Vertex<double> >();
    }

    if(!vectorBorderPoints.size())
    {
        cerr<< "No points received. Aborting Tesselation." << endl;
        return ; //vector<Vertex<double> >();
    }


	/* Begin definition of the polygon to be tesselated */
	gluTessBeginPolygon(m_tesselator, vectorBorderPoints[0].top());
    
    for(int i=0; i<vectorBorderPoints.size(); ++i)
    {
        stack<HVertex*> borderPoints = vectorBorderPoints[i];
        stringstream tFileName; tFileName << "tess__contour_" << i << "__"<< borderPoints.size() << ".txt";
        ofstream orgContour(tFileName.str().c_str());
        HVertex* contourBegin = borderPoints.top();

        // Begin Contour
        gluTessBeginContour(m_tesselator);
        
        /* define the contour by vertices */
        while(borderPoints.size() > 0)
        {
            GLdouble* vertex = new GLdouble[3];
            vertex[0] = (borderPoints.top())->m_position[0];
            vertex[1] = (borderPoints.top())->m_position[1];
            vertex[2] = (borderPoints.top())->m_position[2];
            orgContour << (borderPoints.top())->m_position[0] << " " <<  (borderPoints.top())->m_position[1] << " " << (borderPoints.top())->m_position[2] << endl;
            borderPoints.pop();
            /* Add the vertex to the Contour */
            gluTessVertex(m_tesselator, vertex, borderPoints.top());
        }

        /* End Contour */
        gluTessEndContour(m_tesselator);
        orgContour << contourBegin->m_position[0] << " " << contourBegin->m_position[1] << " " << contourBegin->m_position[2];
        orgContour.close();
    }

 /* End Tesselation */
 gluTessEndPolygon(m_tesselator);
 m_numContours++;
 m_tesselated = true;
 gluDeleteTess(m_tesselator);
 return ;//m_triangles;
}

template<typename VertexT, typename NormalT>
//vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::tesselate(Region<VertexT, NormalT> region)
void Tesselator<VertexT, NormalT>::tesselate(Region<VertexT, NormalT> region)
{
    m_region = region.m_region_number;
    m_normal = region.m_normal;
    //return tesselate(region.getContours(1.1));
    //return 
    tesselate(region.getContours(1.1));
}

} /* namespace lssr */
#endif /* TESSELATOR_C */

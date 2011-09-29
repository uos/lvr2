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
template<typename VertexT, typename NormalT> NormalT Tesselator<VertexT, NormalT>::m_normal;
template<typename VertexT, typename NormalT> int     Tesselator<VertexT, NormalT>::m_numContours=0;
template<typename VertexT, typename NormalT> bool    Tesselator<VertexT, NormalT>::m_debug = true; 
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
    if(m_vertices.size() < 3)
    {
        cout << "Only 3 Points after retriangulation. Aborting.\n";
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
    newVertex.m_normal = m_normal;
    m_vertices.push_back(newVertex);
}

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::getFinalizedTriangles(float **vertexBuffer,
                                                         unsigned int    **indexBuffer,
                                                         int    *lengthFaces,
                                                         int    *lengthVertices)
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
    int numFaces = m_triangles.size() / 3;
    int numVertices = m_triangles.size();

    // allocate new memory.
    (*vertexBuffer) = new float[numVertices*3];
    //(*normalBuffer) = new float[numVertices*3];
    //(*colorBuffer)  = new float[numVertices*3];
    (*indexBuffer) =  new unsigned int[numVertices];


    // init memory
    for(int i=0; i<numVertices*3; ++i)
    {
        (*vertexBuffer)[i] = 0.0;
        //(*normalBuffer)[i] = 0.0;
        //(*colorBuffer)[i]  = 0.0;
    }
    for(int i=0; i<numVertices; ++i)
    {
        (*indexBuffer)[i] = 0.0;
    }


    int usedVertices=0, usedNormals=0, usedColors=0, usedFaces=0;
    int maxColorBufferValue=0;
    
    // keep track of already used vertices to avoid floats.t
    vector<Vertex<float> > vertices;
    vector<Vertex<float> >::iterator triangles    = m_triangles.begin();
    vector<Vertex<float> >::iterator trianglesEnd = m_triangles.end();

    int count=0;
    int posArr[3]; posArr[0]=-1; posArr[1]=-1; posArr[2]=-1;
    // add all triangles and so faces to our buffers and keep track of all used parameters
    int m=0;
    double r,g,b;
    int surface_class = m_region;

    r = fabs(cos(surface_class)); 
    g = fabs(sin(surface_class * 30));
    b = fabs(sin(surface_class * 2));
    for(; triangles != trianglesEnd; ++triangles)
    {
        // try to find the new triangleVertex in the list of used vertices.
        
        /*vector<Vertex<float> >::iterator it    = vertices.begin();
        vector<Vertex<float> >::iterator itEnd = vertices.end();
        int pos=0;
        while(it != itEnd && *it != *triangles) 
        {
            it++;
            pos++;
        }
        if(it != itEnd)
        {
            posArr[m] = pos;
        } else
        { */
			  // vertex was not used before so store it
			  vertices.push_back(*triangles);
			  
			  (*vertexBuffer)[(usedVertices * 3) + 0] = (*triangles)[0]; 
			  (*vertexBuffer)[(usedVertices * 3) + 1] = (*triangles)[1];
			  (*vertexBuffer)[(usedVertices * 3) + 2] = (*triangles)[2];

			  //(*normalBuffer)[(usedVertices * 3) + 0] = m_normal[0];
			  //(*normalBuffer)[(usedVertices * 3) + 1] = m_normal[1];
			  //(*normalBuffer)[(usedVertices * 3) + 2] = m_normal[2];
			  
			  //(*colorBuffer)[(usedVertices *3) + 0] = r;
			  //(*colorBuffer)[(usedVertices *3) + 1] = g;
			  //(*colorBuffer)[(usedVertices *3) + 2] = b;

			  posArr[m] = usedVertices;
			  usedVertices++;
        //}
        m++;
        
        if(m == 3) // we added 3 vertices therefore a whole face!!
        {
            (*indexBuffer)[(usedFaces * 3) + 0] = posArr[0]; 
            (*indexBuffer)[(usedFaces * 3) + 1] = posArr[1];
            (*indexBuffer)[(usedFaces * 3) + 2] = posArr[2];
            /* check for corrupt vertices! */
            float x1 = (*vertexBuffer)[posArr[0]];
            float y1 = (*vertexBuffer)[posArr[0]+1];
            float z1 = (*vertexBuffer)[posArr[0]+2]; 

            float x2 = (*vertexBuffer)[posArr[1]];
            float y2 = (*vertexBuffer)[posArr[1]+1];
            float z2 = (*vertexBuffer)[posArr[1]+2]; 

            float x3 = (*vertexBuffer)[posArr[2]];
            float y3 = (*vertexBuffer)[posArr[2]+1];
            float z3 = (*vertexBuffer)[posArr[2]+2]; 

            float d12 = sqrt( pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2) );
            float d13 = sqrt( pow((x1-x3),2) + pow((y1-y3),2) + pow((z1-z3),2) );
            float d23 = sqrt( pow((x2-x3),2) + pow((y2-y3),2) + pow((z2-z3),2) );
            if( d12 <= 0.001 || d13 <= 0.001 || d23 <= 0.001){
                cout << "Damnit DEAD Face!: ";
                cout << "positions: " << posArr[0] << " " << posArr[1] << " " << posArr[2] << endl;
            }
				#ifdef DB_TESS
            cout << "v1: " << (*vertexBuffer)[posArr[0]] << " " << (*vertexBuffer)[posArr[0]+1] << " " << (*vertexBuffer)[posArr[0]+2] << "\n"; 
            cout << "v2: " << (*vertexBuffer)[posArr[1]] << " " << (*vertexBuffer)[posArr[1]+1] << " " << (*vertexBuffer)[posArr[1]+2] << "\n"; 
            cout << "v3: " << (*vertexBuffer)[posArr[2]] << " " << (*vertexBuffer)[posArr[2]+1] << " " << (*vertexBuffer)[posArr[2]+2] << "\n"; 
            cout << "positions: " << posArr[0] << " " << posArr[1] << " " << posArr[2] << endl;
				#endif
            usedFaces++;
            m=0;
        }
    }

    if(usedFaces > 0 && usedVertices > 0)
    {
        // Copy all that stuff and resize array -- this should be improved somehow! TODO:!
        float *newVertexBuffer = new float[usedVertices*3];
        float *newNormalBuffer = new float[usedVertices*3];
        float *newColorBuffer  = new float[usedVertices*3];
        unsigned int    *newIndexBuffer  = new unsigned int[usedFaces*3];

        // use memcopy?
        for(int i=0; i<usedVertices*3; i++)
        {
            newVertexBuffer[i] = (*vertexBuffer)[i];
            //newNormalBuffer[i] = (*normalBuffer)[i];
            //newColorBuffer[i]  = (*colorBuffer)[i];
        }

        for(int i=0; i<usedFaces*3; ++i)
        {
            newIndexBuffer[i] = (*indexBuffer)[i];
        }
        //delete (*colorBuffer);
        delete (*indexBuffer);
        delete (*vertexBuffer);
        //delete (*normalBuffer);

        (*vertexBuffer) = newVertexBuffer;
        //(*normalBuffer) = newNormalBuffer;
        //(*colorBuffer)  = newColorBuffer;
        (*indexBuffer) = newIndexBuffer;
    }
    *lengthVertices = usedVertices*3;
    *lengthFaces = usedFaces*3;
    #ifdef DB_TESS
    cout << "Retesselation Complete. " << usedVertices << " Vertices. " << usedFaces << " Faces.\n";
    #endif
}


template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::tesselatorCombineVertices(GLdouble coords[3],
							 GLdouble *vertex_data[4],
							 GLdouble weight[4],
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
    m_vertices.push_back(HVertex(v));
	*dataOut = vertex;
}

template<typename VertexT, typename NormalT>
void Tesselator<VertexT, NormalT>::init(void)
{
    m_vertices.clear();
    m_triangles.clear();
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
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_POSITIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NEGATIVE);
    //gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
    gluTessProperty(m_tesselator, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD);
	 /* Use gluTessNormal: speeds up the tessellation if the
	  	Polygon lies on a x-y plane. and it approximatly does!*/
	 //gluTessNormal(m_tesselator, 0, 0, 1);
}

template<typename VertexT, typename NormalT>
//vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::tesselate(vector<stack<HVertex*> > vectorBorderPoints)
void Tesselator<VertexT, NormalT>::tesselate(vector<vector<HVertex*> > vectorBorderPoints)
{
    if(!m_tesselator)
    {
        cerr<<"No Tesselation Object Created. Please use Tesselator::init() before making use of Tesselator::tesselate(...). Aborting Tesselation." << endl;
        return ; //vector<Vertex<double> >();
    }

    if(!vectorBorderPoints.size())
    {
        //cerr<< "No points received. Aborting Tesselation." << endl;
        return ; //vector<Vertex<double> >();
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
        stringstream tFileName; tFileName << "region[" << setw(5) << setfill('0') 
                                          <<m_region << "]_contour[" << 
                                          setw(3) << setfill ('0') << i << "]_size["<< borderPoints.size() << "].txt";
        ofstream orgContour(tFileName.str().c_str());
        #endif

        if(borderPoints.size() <3 )
        {
            #ifdef DB_TESS
            cout << "BorderContains less than 3 Points!. Aborting.\n";
				cout << "i: " << i << " size:" << borderPoints.size() << endl;
            orgContour.close();
            //remove(tFileName.str().c_str());
            #endif
            continue; 
        }
        
        HVertex* contourBegin = borderPoints.back();
        
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
 m_tesselated = true;
 gluDeleteTess(m_tesselator);
 m_tesselator = 0;
 return;
}

template<typename VertexT, typename NormalT>
//vector<HalfEdgeVertex<VertexT, NormalT> > Tesselator<VertexT, NormalT>::tesselate(Region<VertexT, NormalT> region)
void Tesselator<VertexT, NormalT>::tesselate(Region<VertexT, NormalT> *region)
{
    m_region = region->m_region_number;
    m_normal = region->calcNormal(); //.m_normal;
    tesselate(region->getContours(0.01));
    //cout << "Anzahl Contouren nach kopie: " << region->getContours(0.01).size() << endl;
}

} /* namespace lssr */
#endif /* TESSELATOR_C */


/** DCC:
	 for(int i=0; i<nPointsUsed-2; i++)
    {
        cout << "Vertex: ";
        cout << this->m_vertexBuffer[i] << " ";
        cout << this->m_vertexBuffer[i+1] << " ";
        cout << this->m_vertexBuffer[i+2] << " " << endl;
        cout << "Color: ";
        cout << this->m_colorBuffer[i] << " ";
        cout << this->m_colorBuffer[i+1] << " ";
        cout << this->m_colorBuffer[i+2] << " " << endl;
        cout << "Normal: ";
        cout << this->m_normalBuffer[i] << " ";
        cout << this->m_normalBuffer[i+1] << " ";
        cout << this->m_normalBuffer[i+2] << " " << endl;
    }
    cout << "POOOINTS : " << nPointsUsed << endl;
    for(int i=0; i<nIndizesUsed; i++)
    {
        cout << "Index: ";
        cout << this->m_indexBuffer[i] << " " << endl;
    }
    cout << "FIIIIIIILE" << endl;
    for(int i=0; i<nIndizesUsed; i++)
    {
        cout << this->m_vertexBuffer[this->m_indexBuffer[i] *3 + 0] << " " << this->m_indexBuffer[i] *3 + 0 << " ";
        cout << this->m_vertexBuffer[this->m_indexBuffer[i] *3 + 1] << " " << this->m_indexBuffer[i] *3 + 1 << " ";
        cout << this->m_vertexBuffer[this->m_indexBuffer[i] *3 + 2] << " " << this->m_indexBuffer[i] *3 + 2 << " ";
        cout << endl;
    }

  */

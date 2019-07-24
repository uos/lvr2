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
 * StaticMesh.cpp
 *
 *  Created on: 12.11.2008
 *      Author: Thomas Wiemann
 */

#include "lvr2/display/StaticMesh.hpp"

#include <cassert>

namespace lvr2
{

StaticMesh::StaticMesh(){

	m_normals.reset();
	m_faceNormals   = 0;
    m_blackColors   = 0;
	m_vertices.reset();
	m_colors.reset();
	m_faces.reset();

	m_numFaces      = 0;
	m_numVertices   = 0;
	m_numMaterials  = 0;

	m_finalized = false;
	m_haveMaterials = false;

	m_renderMode    = 0;

	m_nameList = -1;

}

StaticMesh::StaticMesh( ModelPtr model, string name )
    : Renderable( name )
{

    m_model = model;
    m_nameList = -1;
    m_blackColors   = 0;

    init( model->m_mesh );
	calcBoundingBox();
	compileColoredMeshList();
	compileWireframeList();
	compileNameList();

}

StaticMesh::StaticMesh( MeshBufferPtr mesh, string name )
    : Renderable( name )
{

    m_model = ModelPtr( new Model( mesh ) );
    m_nameList = -1;
    m_blackColors   = 0;
    init( mesh );

    calcBoundingBox();
    compileColoredMeshList();
    compileWireframeList();
    compileNameList();

}

void StaticMesh::init( MeshBufferPtr mesh )
{
	m_lineWidth = 2.0;
	if(mesh)
	{
		m_faceNormals = 0;
        m_numVertices = mesh->numVertices();
        m_numFaces = mesh->numFaces();
        size_t w_color;

		m_normals 			= mesh->getVertexNormals();
		m_colors        	= mesh->getVertexColors(w_color);

        // does only support RGB vertex colors
        assert(w_color == 3);

		m_vertices      	= mesh->getVertices();
		m_faces       		= mesh->getFaceIndices();
		m_blackColors		= new unsigned char[ 3 * m_numVertices ];

		for ( size_t i = 0; i < 3 * m_numVertices; i++ ) 
		{
			m_blackColors[i] = 0;
		}

		m_finalized			= true;
		m_visible			= true;
		m_active			= true;

		m_renderMode = 0;
		m_renderMode    |= RenderSurfaces;
		m_renderMode    |= RenderTriangles;

		m_boundingBox = new BoundingBox<Vec>;

		if(!m_faceNormals)
		{
			interpolateNormals();
		} else
		{
			// cout << "Face normals: " << m_faceNormals << endl;
		}

		if(!m_colors)
		{
			setDefaultColors();
		}
	}
}


StaticMesh::StaticMesh(const StaticMesh &o)
{
	if( m_faceNormals != 0 )
	  {
	    delete[] m_faceNormals;
	  }

	m_faceNormals = new float[3 * o.m_numVertices];
	m_vertices    = floatArr( new float[3 * o.m_numVertices] );
	m_colors      = ucharArr( new unsigned char[3 * o.m_numVertices] );
	m_faces     = indexArray(  new unsigned int[3 * o.m_numFaces] );

	for ( size_t i(0); i < 3 * o.m_numVertices; i++ )
	{
		m_faceNormals[i] = o.m_faceNormals[i];
		m_vertices[i]    = o.m_vertices[i];
		m_colors[i]      = o.m_colors[i];
	}

	for( size_t i = 0; i < 3 * o.m_numFaces; ++i )
	{
		m_faces[i] = o.m_faces[i];
	}

	m_boundingBox = o.m_boundingBox;
	m_model = o.m_model;

}

void StaticMesh::setColorMaterial(float r, float g, float b)
{
	float ambient_color[] = {r, g, b};
	float diffuse_color[] = {0.45f * r, 0.5f * g, 0.55f * b};

	float specular_color[] = {0.1f, 0.15f, 0.1f};
	float shine[] = {0.1f};

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient_color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse_color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shine);
}


StaticMesh::~StaticMesh(){
    if (m_blackColors)
        delete[] m_blackColors;

    if (m_faceNormals)
        delete[] m_faceNormals;
}

void StaticMesh::finalize(){

}

void StaticMesh::compileWireframeList()
{
    if(m_finalized){

        m_wireframeList = glGenLists(1);

        // Start new display list
        glNewList(m_wireframeList, GL_COMPILE);

        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glColor3f(0.0, 0.0, 0.0);

        for(size_t i = 0; i < m_numFaces; i++)
        {
            size_t index = 3 * i;
            int a = 3 * m_faces[index];
            int b = 3 * m_faces[index + 1];
            int c = 3 * m_faces[index + 2];
            glBegin(GL_TRIANGLES);
            glVertex3f(m_vertices[a], m_vertices[a + 1], m_vertices[a + 2]);
            glVertex3f(m_vertices[b], m_vertices[b + 1], m_vertices[b + 2]);
            glVertex3f(m_vertices[c], m_vertices[c + 1], m_vertices[c + 2]);
            glEnd();

        }
        glEnable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glEndList();

    }
}


void StaticMesh::compileColoredMeshList(){

	if(m_finalized){

		m_coloredMeshList = glGenLists(1);

		// Enable vertex / normal / color arrays
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		// Start new display list
		glNewList(m_coloredMeshList, GL_COMPILE);

		glEnable(GL_LIGHTING);
		glEnable(GL_COLOR_MATERIAL);

		setColorMaterial(0.1f, 0.1f, 0.1f);

		// Assign element pointers
		glVertexPointer( 3, GL_FLOAT, 0, m_vertices.get() );
		glNormalPointer( GL_FLOAT, 0, m_faceNormals );
		glColorPointer( 3, GL_UNSIGNED_BYTE, 0, m_colors.get() );

		// Draw elements
		glDrawElements(GL_TRIANGLES, (GLsizei)3 * m_numFaces, GL_UNSIGNED_INT, m_faces.get());


		// Draw mesh descriptions


		glEndList();

	}
}

void StaticMesh::setName(string name)
{
	m_name = name;
	compileNameList();
}

void StaticMesh::compileNameList()
{
	// Release old name list
	if(m_nameList != -1)
	{
		glDeleteLists(m_nameList, 1);
	}

	// Compile a new one
	m_nameList = glGenLists(1);
	glNewList(m_nameList, GL_COMPILE);
	Vec v = m_boundingBox->getCentroid();
	glDisable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 0.0);
	glRasterPos3f(v.x, v.y, v.z);
	for(int i = 0; i < Name().size(); i++)
	{
		//glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, Name()[i]);
	}
	glEnable(GL_LIGHTING);
	glEndList();
}

void StaticMesh::interpolateNormals()
{

	// Be sure that vertex and indexbuffer exist
	assert(m_vertices);
	assert(m_faces);

	// Alloc new normal array
	m_faceNormals = new float[3 * m_numVertices];
	memset(m_faceNormals, 0, 3 * m_numVertices * sizeof(float));

	// Interpolate surface m_normals for each face
	// and interpolate sum up the normal coordinates
	// at each vertex position
	size_t a, b, c, buffer_pos;
	for(size_t i = 0; i < m_numFaces; i++)
	{
		buffer_pos = i * 3;

		// Interpolate a perpendicular vector to the
		// current triangle (p)
		//
		// CAUTION:
		// --------
		// buffer_pos is the face number
		// to get real position of the vertex in the buffer
		// we have to remember, that each vertex has three
		// coordinates!		cout << 1 << endl;
		a = m_faces[buffer_pos]     * 3;
		b = m_faces[buffer_pos + 1] * 3;
		c = m_faces[buffer_pos + 2] * 3;

		Vec v0(m_vertices[a], m_vertices[a + 1], m_vertices[a + 2]);
		Vec v1(m_vertices[b], m_vertices[b + 1], m_vertices[b + 2]);
		Vec v2(m_vertices[c], m_vertices[c + 1], m_vertices[c + 2]);

		Vec d1 = v0 - v1;
		Vec d2 = v2 - v1;

		Normal<float> p(d1.cross(d2));
		p = p * -1;

		// Sum up coordinate values in normal array
		m_faceNormals[a    ] = p.x;
		m_faceNormals[a + 1] = p.y;
		m_faceNormals[a + 2] = p.z;

		m_faceNormals[b    ] = p.x;
		m_faceNormals[b + 1] = p.y;
		m_faceNormals[b + 2] = p.z;

		m_faceNormals[c    ] = p.x;
		m_faceNormals[c + 1] = p.y;
		m_faceNormals[c + 2] = p.z;

	}

	// Normalize
	for(size_t i = 0; i < m_numVertices; i++)
	{
		Normal<float> n(m_faceNormals[i * 3], m_faceNormals[i * 3 + 1], m_faceNormals[i * 3 + 2]);
		m_faceNormals[i * 3]     = n.x;
		m_faceNormals[i * 3 + 1] = n.y;
		m_faceNormals[i * 3 + 2] = n.z;
	}

}

void StaticMesh::setDefaultColors()
{
    m_colors = ucharArr( new unsigned char[3 * m_numVertices] );
    for(size_t i = 0; i < m_numVertices; i++)
    {
        m_colors[i*3 + 0] = 0;
        m_colors[i*3 + 1] = 255;
        m_colors[i*3 + 2] = 0;
    }
}

void StaticMesh::calcBoundingBox()
{
    for(size_t i = 0; i < m_numVertices; i++)
    {
        m_boundingBox->expand(Vec(
                m_vertices[3 * i],
                m_vertices[3 * i + 1],
                m_vertices[3 * i + 2] ));

    }
}

uintArr StaticMesh::getIndices()
{

    return m_finalized ? m_faces : uintArr();

}

floatArr StaticMesh::getVertices()
{

    return m_finalized ? m_vertices : floatArr();

}

float* StaticMesh::getNormals()
{

    return m_finalized ? m_faceNormals : 0;

}

size_t StaticMesh::getNumberOfVertices()
{
	return m_numVertices;
}
size_t StaticMesh::getNumberOfFaces()
{
	return m_numFaces;
}

void StaticMesh::savePLY(string filename)
{

}


} // namespace lvr2

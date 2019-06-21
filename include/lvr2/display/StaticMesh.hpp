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
 * StaticMesh.h
 *
 *  Created on: 12.11.2008
 *      Author: Thomas Wiemann
 */

#ifndef STATICMESH_H_
#define STATICMESH_H_

#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <stdlib.h>
#include <string>
#include <string.h>
#include <sstream>

#ifdef _MSC_VER
#include <Windows.h>
#endif

#ifndef __APPLE__
#include <GL/gl.h>
#else
#include <OpenGL/gl.h>
#endif

#include "lvr2/display/Renderable.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

using namespace std;


namespace lvr2
{

enum
{
    RenderSurfaces            = 0x01,
    RenderTriangles           = 0x02,
    RenderSurfaceNormals      = 0x04,
    RenderVertexNormals       = 0x08,
    RenderColors              = 0x10,
};

class StaticMesh : public Renderable
{
public:
	StaticMesh();
	StaticMesh( ModelPtr model, string name=""); // <unnamed static mesh>" );
	StaticMesh( MeshBufferPtr buffer, string name=""); //<unnamed static mesh>" );

	StaticMesh(const StaticMesh &o);
	~StaticMesh();
	inline void render();

	virtual void finalize();
	virtual void savePLY(string filename);
	virtual void setName(string name);

	floatArr		getVertices();
	uintArr 		getIndices();
	float*          getNormals();

	size_t			getNumberOfVertices();
	size_t			getNumberOfFaces();

	void setRenderMode(int mode) { m_renderMode = mode;}
	int  getRenderMode() { return m_renderMode;}

private:

	void init( MeshBufferPtr mesh );

	void interpolateNormals();
	void setDefaultColors();
	void calcBoundingBox();

protected:

	//void compileDisplayLists();
	void compileColoredMeshList();
	void compileWireframeList();
	void compileNameList();

	void setColorMaterial(float r, float g, float b);


	void readPly(string filename);

	floatArr        m_normals;
	float*          m_faceNormals;
	floatArr        m_vertices;
	ucharArr        m_colors;
	unsigned char*  m_blackColors;

	indexArray      m_faces;

	bool            m_finalized;
	bool			m_haveMaterials;

	size_t          m_numVertices;
	size_t          m_numFaces;
	size_t          m_numMaterials;

	int             m_renderMode;


	int             m_coloredMeshList;
	int             m_wireframeList;
	int				m_nameList;

};

void StaticMesh::render(){

	if(m_active)
	{
		if(m_finalized){
		    glPushMatrix();
		    glMultMatrixf(m_transformation.getData());
		    if(m_renderMode & RenderSurfaces)
		    {
		        glEnable(GL_LIGHTING);
		        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		        glCallList(m_coloredMeshList);
		        glCallList(m_nameList);
		    }

		    if(m_renderMode & RenderTriangles)
		    {
		        glDisable(GL_LIGHTING);
		        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		        glLineWidth(m_lineWidth);
		        glColor3f(0.0, 0.0, 0.0);
		        glCallList(m_wireframeList);
		        glEnable(GL_LIGHTING);
		    }
		    glPopMatrix();

 		}
	}
}

} // namespace lvr2

#endif /* STATICMESH_H_ */

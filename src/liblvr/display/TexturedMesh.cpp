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
 * TexturedMesh.cpp
 *
 *  Created on: 14.04.2012
 *      Author: Thomas Wiemann
 */

#include "display/TexturedMesh.hpp"

#include <map>
using std::multimap;
using std::map;

namespace lvr
{

void TexturedMesh::getBufferArray(unsigned int* buffer, MaterialGroup* g)
{
	size_t n = g->faceBuffer.size();
	for(size_t i = 0; i < n; i++)
	{
		size_t pos = 3 * i;

		buffer[pos    ] = m_faces[3 * g->faceBuffer[i]    ];
		buffer[pos + 1] = m_faces[3 * g->faceBuffer[i] + 1];
		buffer[pos + 2] = m_faces[3 * g->faceBuffer[i] + 2];
	}

}

TexturedMesh::TexturedMesh(MeshBufferPtr mesh)
{
	// Store internal buffers
	size_t	dummy;
	m_faces 		= mesh->getFaceArray(m_numFaces);
	m_faceMaterials = mesh->getFaceMaterialIndexArray(dummy);
	m_vertices 		= mesh->getVertexArray(m_numVertices);
	m_normals		= mesh->getVertexNormalArray(dummy);
	m_texcoords		= mesh->getVertexTextureCoordinateArray(dummy);
	m_textures		= mesh->getTextureArray(m_numTextures);
	m_materials		= mesh->getMaterialArray(m_numMaterials);

	// Calc bounding box
	for(size_t i = 0; i < m_numVertices; i++)
	{
		float x = m_vertices[3 * i];
		float y = m_vertices[3 * i + 1];
		float z = m_vertices[3 * i + 2];
		m_boundingBox->expand(x, y, z);
	}

	// Create material groups for optimized rendering
	generateMaterialGroups();

	// Compile display list
	compileTexureDisplayList();
	compileWireframeList();

	m_finalized = true;

	m_renderMode = 0;
	m_renderMode    |= RenderSurfaces;
	m_renderMode    |= RenderTriangles;
}


void TexturedMesh::generateMaterialGroups()
{

	map<int, MaterialGroup* > texMatMap;
	map<Vertex<unsigned char>, MaterialGroup* > colorMatMap;

	// Iterate over face material buffer and
	// sort faces by their material
	for(size_t i = 0; i < m_numFaces; i++)
	{
		map<int, MaterialGroup*>::iterator texIt;
		map<Vertex<unsigned char>, MaterialGroup* >::iterator colIt;

		// Get material by index and lookup in map. If present
		// add face index to the corresponding group. Create a new
		// group if none was found. For efficient rendering we have to
		// create groups by color and texture index,
		Material* m = m_materials[m_faceMaterials[i]];

		if(m->texture_index != -1)
		{

			texIt = texMatMap.find(m->texture_index);
			if(texIt == texMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m->texture_index;
				g->color = Vertex<float>(1.0, 1.0, 1.0);
				g->faceBuffer.push_back(i);
				m_textureMaterials.push_back(g);
				texMatMap[m->texture_index] = g;
			}
			else
			{
				texIt->second->faceBuffer.push_back(i);
			}
		}
		else
		{
			colIt = colorMatMap.find(Vertex<unsigned char>(m->r, m->g, m->b));
			if(colIt == colorMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m->texture_index;
				g->faceBuffer.push_back(i);
				g->color = Vertex<float>(m->r / 255.0f, m->g / 255.0f, m->b / 255.0f);
				m_colorMaterials.push_back(g);
			}
			else
			{
				colIt->second->faceBuffer.push_back(i);
			}
		}
	}
}

void TexturedMesh::compileTexureDisplayList()
{

	// Enable vertex arrays
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);


	// Get display list index and compile display list
	m_textureDisplayList = glGenLists(1);


	glNewList(m_textureDisplayList, GL_COMPILE);

	// Bind arrays
	glVertexPointer( 3, GL_FLOAT, 0, m_vertices.get() );
	glNormalPointer( GL_FLOAT, 0, m_normals.get() );
	glTexCoordPointer(3, GL_FLOAT, 0, m_texcoords.get());


	// Draw textured materials
	for(size_t i = 0; i < m_textureMaterials.size(); i++)
	{
		MaterialGroup* g = m_textureMaterials[i];

		unsigned int* buf = new unsigned int [3 * g->faceBuffer.size()];
		getBufferArray(buf, g);
		m_textures[g->textureIndex]->bind();
		glColor3f(1.0, 1.0, 1.0);
		setColorMaterial(g->color[0], g->color[1], g->color[2]);
		glDrawElements(GL_TRIANGLES, 3 * g->faceBuffer.size(), GL_UNSIGNED_INT, buf);
	}

	// Draw colored materials

	glDisable(GL_TEXTURE_2D);
	glBegin(GL_TRIANGLES);
	for(size_t i = 0; i < m_colorMaterials.size(); i++)
	{
		MaterialGroup* g = m_colorMaterials[i];
		glColor3f(g->color[0], g->color[1], g->color[2]);
		for(size_t i = 0; i < g->faceBuffer.size(); i++)
		{
			int a = m_faces[g->faceBuffer[i] * 3    ];
			int b = m_faces[g->faceBuffer[i] * 3 + 1];
			int c = m_faces[g->faceBuffer[i] * 3 + 2];


			glNormal3f(m_normals	[3 * a], m_normals	[3 * a + 1], m_normals	[3 * a + 2]);
			glVertex3f(m_vertices	[3 * a], m_vertices	[3 * a + 1], m_vertices	[3 * a + 2]);


			glNormal3f(m_normals	[3 * b], m_normals	[3 * b + 1], m_normals	[3 * b + 2]);
			glVertex3f(m_vertices	[3 * b], m_vertices	[3 * b + 1], m_vertices	[3 * b + 2]);

			glNormal3f(m_normals	[3 * c], m_normals	[3 * c + 1], m_normals	[3 * c + 2]);
			glVertex3f(m_vertices	[3 * c], m_vertices	[3 * c + 1], m_vertices	[3 * c + 2]);
		}

	}
	glEnd();
	glEnable(GL_TEXTURE_2D);
	glEndList();

}



}




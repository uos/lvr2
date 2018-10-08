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


#include <lvr2/display/TexturedMesh.hpp>

#include <map>
using std::multimap;
using std::map;

namespace lvr2
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

TexturedMesh::TexturedMesh(MeshBuffer2Ptr mesh) : m_materials(mesh->getMaterials())
{
	// Store internal buffers
    m_numFaces = mesh->numFaces();
    m_numVertices = mesh->numVertices();
    m_numTextures = mesh->getTextures().size();
    m_numMaterials = mesh->getMaterials().size();

	m_faces 		= mesh->getFaceIndices();
	m_faceMaterials = mesh->getFaceMaterialIndices();
	m_vertices 		= mesh->getVertices();
	m_normals		= mesh->getVertexNormals();
	m_texcoords		= mesh->getTextureCoordinates();
    m_boundingBox   = new BoundingBox<Vec>;

    // convert to GlTexture*
    m_textures      = textureArr( new GlTexture* [mesh->getTextures().size()] );
    for (size_t i = 0; i < mesh->getTextures().size(); i++)
    {
        m_textures[i] = new GlTexture(mesh->getTextures()[i]);
    }

	// Calc bounding box
	for(size_t i = 0; i < m_numVertices; i++)
	{
		float x = m_vertices[3 * i];
		float y = m_vertices[3 * i + 1];
		float z = m_vertices[3 * i + 2];
		m_boundingBox->expand(Vector<Vec>(x, y, z));
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
    using VecUChar = BaseVector<unsigned char>;
    using compColorsT = std::function<bool (const Vector<VecUChar> &, const Vector<VecUChar> &)>;
    compColorsT colorsCompare = [] (const Vector<VecUChar> &a, const Vector<VecUChar> &b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z); };

	map<int, MaterialGroup* > texMatMap;
	map<Vector<VecUChar>, MaterialGroup*, compColorsT> colorMatMap;

	// Iterate over face material buffer and
	// sort faces by their material
	for(size_t i = 0; i < m_numFaces; i++)
	{
		map<int, MaterialGroup*>::iterator texIt;
		map<Vector<VecUChar>, MaterialGroup*, compColorsT>::iterator colIt;

		// Get material by index and lookup in map. If present
		// add face index to the corresponding group. Create a new
		// group if none was found. For efficient rendering we have to
		// create groups by color and texture index,
        Material& m = m_materials[m_faceMaterials[i]];

		if(m.m_texture)
		{

			texIt = texMatMap.find(m.m_texture->idx());
			if(texIt == texMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = m.m_texture->idx();
				g->color = Vector<Vec>(1.0, 1.0, 1.0);
				g->faceBuffer.push_back(i);
				m_textureMaterials.push_back(g);
				texMatMap[m.m_texture->idx()] = g;
			}
			else
			{
				texIt->second->faceBuffer.push_back(i);
			}
		}
		else
		{
			colIt = colorMatMap.find(Vector<VecUChar>(m.m_color->at(0), m.m_color->at(1), m.m_color->at(2)));
			if(colIt == colorMatMap.end())
			{
				MaterialGroup* g = new MaterialGroup;
				g->textureIndex = -1;
				g->faceBuffer.push_back(i);
				g->color = Vector<Vec>(m.m_color->at(0) / 255.0f, m.m_color->at(0) / 255.0f, m.m_color->at(0) / 255.0f);
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
        delete[] buf;
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



} // namespace lvr2

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
 * TexturedMesh.hpp
 *
 *  Created on: 14.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef TEXTUREDMESH_HPP_
#define TEXTUREDMESH_HPP_

#include "Renderable.hpp"

namespace lssr
{

struct MaterialGroup
{
	int	textureIndex;
	int numFaces;
	Vertex<float> color;
	vector<unsigned int> faceBuffer;
};

class TexturedMesh: public lssr::Renderable
{
public:
	TexturedMesh(MeshBufferPtr mesh);
	virtual ~TexturedMesh() {};
	virtual void render()
	{
		if(m_active)
		{
			glCallList(m_displayList);
		}
	}


private:

	void setColorMaterial(float r, float g, float b);
	void generateMaterialGroups();
	void compileDisplayList();
	void getBufferArray(unsigned int*, MaterialGroup* g);

	uintArr					m_faces;
	uintArr					m_faceMaterials;
	floatArr				m_texcoords;
	floatArr				m_normals;
	floatArr				m_vertices;
	materialArr				m_materials;
	textureArr 				m_textures;


	vector<MaterialGroup*> 	m_textureMaterials;
	vector<MaterialGroup*> 	m_colorMaterials;

	size_t					m_numFaces;
	size_t					m_numMaterials;
	size_t					m_numTextures;
	size_t					m_numVertices;

	int						m_displayList;
};

} /* namespace lssr */
#endif /* TEXTUREDMESH_HPP_ */

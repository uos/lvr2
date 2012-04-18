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

#include "StaticMesh.hpp"

namespace lssr
{

struct MaterialGroup
{
	int	textureIndex;
	int numFaces;
	Vertex<float> color;
	vector<unsigned int> faceBuffer;
};

class TexturedMesh: public StaticMesh
{
public:
	TexturedMesh(MeshBufferPtr mesh);
	virtual ~TexturedMesh() {};


	virtual void render()
	{
		if(m_active)
		{
			if(m_finalized){
				glPushMatrix();
				glMultMatrixf(m_transformation.getData());
				if(m_renderMode & RenderSurfaces)
				{
					//glEnable(GL_LIGHTING);
					//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					glCallList(m_textureDisplayList);
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


private:

	void generateMaterialGroups();
	void compileTexureDisplayList();
	void getBufferArray(unsigned int*, MaterialGroup* g);

	uintArr					m_faceMaterials;
	floatArr				m_texcoords;
	materialArr				m_materials;
	textureArr 				m_textures;


	vector<MaterialGroup*> 	m_textureMaterials;
	vector<MaterialGroup*> 	m_colorMaterials;

	size_t					m_numFaces;
	size_t					m_numMaterials;
	size_t					m_numTextures;
	size_t					m_numVertices;

	int						m_textureDisplayList;
};

} /* namespace lssr */
#endif /* TEXTUREDMESH_HPP_ */

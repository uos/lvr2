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
 * TexturedMesh.hpp
 *
 *  Created on: 14.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef TEXTUREDMESH_HPP_
#define TEXTUREDMESH_HPP_

#include "lvr2/display/StaticMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"

namespace lvr2
{

using Vec = BaseVector<float>;

struct MaterialGroup
{
	int	textureIndex;
	int numFaces;
	Vec color;
	vector<size_t> faceBuffer;
};

class TexturedMesh: public StaticMesh
{
public:
	TexturedMesh(MeshBufferPtr mesh);

	virtual ~TexturedMesh()
    {
        if (m_textures)
        {
            for (size_t i = 0; i < m_numTextures; i++)
            {
                delete m_textures[i];
            }

        }

        for (MaterialGroup *ptr : m_textureMaterials)
            delete ptr;

        for (MaterialGroup *ptr : m_colorMaterials)
            delete ptr;
    }

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

	indexArray				m_faceMaterials;
	floatArr				m_texcoords;
	textureArr 				m_textures;
	vector<Material>&		m_materials;


	vector<MaterialGroup*> 	m_textureMaterials;
	vector<MaterialGroup*> 	m_colorMaterials;

	size_t					m_numFaces;
	size_t					m_numMaterials;
	size_t					m_numTextures;
	size_t					m_numVertices;

	int						m_textureDisplayList;
};

} /* namespace lvr2 */

#endif /* TEXTUREDMESH_HPP_ */

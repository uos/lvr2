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
 * ObjIO.h
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef OBJIO_HPP_
#define OBJIO_HPP_

namespace lssr
{

/**
 * @brief A basic implementation of the obj file format. Currently
 *        only geometry information is supported. Color / Material
 *        support will follow shortly.
 */
template<typename CoordType, typename IndexType>
class ObjIO
{
public:
    ObjIO();

    void write(string filename);
    void setVertexArray(CoordType* array, size_t count);
    void setNormalArray(CoordType* array, size_t count);
    void setIndexArray(IndexType* array, size_t count);
    void setTextureCoords(CoordType* coords, size_t c);
    void setColors(uchar* coords, size_t c);
    void setTextureIndices(IndexType* coords, size_t c);
    void setTextures(IndexType* textures, size_t c);


private:
	 CoordType*          m_vertices;
	 CoordType*          m_normals;
	 uchar*			     m_colors;
	 CoordType*          m_textureCoords;

	 IndexType*          m_indices;
	 IndexType*				m_textureIndices;
	 IndexType*				m_textures;

	 size_t					m_colorCount;
	 size_t 					m_textureCount;
	 size_t 					m_textureIndicesCount;
	 size_t              m_faceCount;
	 size_t              m_vertexCount;
	 size_t					m_textureCoordsCount;

};

}

#include "ObjIO.tcc"

#endif /* OBJIO_H_ */

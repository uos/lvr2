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
 * TextureToken.hpp
 *
 *  @date 03.05.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef TEXTURETOKEN_HPP_
#define TEXTURETOKEN_HPP_

#include "Texture.hpp"

namespace lvr {

/**
 * @brief	This class allows to calculate texture coordinates for the associated texture.
 */
template<typename VertexT, typename NormalT>
class TextureToken {
public:

	/**
	 * @brief 	Constructor.
	 *
	 * @param 	v1 	The first vector of the texture coordinate system
	 *
	 * @param 	v2 	The second vector of the texture coordinate system
	 *
	 * @param	p	A point in the texture plane
	 *
	 * @param 	a_min	This value defines the bounding box of the texture
	 *
	 * @param 	b_min	This value defines the bounding box of the texture
	 *
	 * @param 	t	The associated texture
	 *
	 * @param 	index	The index of the texture in the texture package
	 *
	**/
	TextureToken(NormalT v1, NormalT v2, VertexT p, float a_min, float b_min, Texture* t = 0, int index = -1);

	/**
	 * @brief	computes texture coordinates corresponding to the give Vertex
	 *
	 * @param	v	the vertex to generate texture coordinates for
	 *
	 * @param	x	returns texture coordinates in x direction
	 *
	 * @param	y	returns texture coordinates in y direction
	 */
	void textureCoords(VertexT v, float &x, float &y);

	/**
	 * @brief 	Destructor.
	 *
         */
	~TextureToken(){/*delete m_texture;*/};

	///The associated texture	
	Texture* m_texture;

	///The coordinate system of the texture plane
	NormalT v1, v2;

	///A point in the texture plane
	VertexT p;

	///The bounding box of the texture plane
	float a_min, b_min;

	///index of the texture in the texture pack
	size_t m_textureIndex;

	///Matrix that stores an affine transform that will be applied to the texture coordinates
	double m_transformationMatrix[6];

	///Indicates if the texture coordinates have to be mirrored or not
	unsigned char m_mirrored;
};

}

#include "TextureToken.tcc"

#endif /* TEXTURETOKEN_HPP_ */

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
 * TextureToken.tcc
 *
 *  @date 03.05.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

namespace lvr {

template<typename VertexT, typename NormalT>
TextureToken<VertexT, NormalT>::TextureToken(NormalT v1, NormalT v2, VertexT p, float a_min, float b_min, Texture* t, int index)
{
	this->v1 	= v1;
	this->v2 	= v2;
	this->p		= p;
	this->a_min	= a_min;
	this->b_min	= b_min;
	this->m_texture	= t;
	this->m_textureIndex = index;
	this->m_transformationMatrix[0] = 1;
	this->m_transformationMatrix[1] = 0;
	this->m_transformationMatrix[2] = 0;
	this->m_transformationMatrix[3] = 0;
	this->m_transformationMatrix[4] = 1;
	this->m_transformationMatrix[5] = 0;
	this->m_mirrored 		= 0;
}

template<typename VertexT, typename NormalT>
void TextureToken<VertexT, NormalT>::textureCoords(VertexT v, float &x, float &y)
{
	VertexT w =  v - ((v1 * a_min) + (v2 * b_min) + p);
	x = (v1 * (w * v1)).length() / Texture::m_texelSize / m_texture->m_width;
	y = (v2 * (w * v2)).length() / Texture::m_texelSize / m_texture->m_height;

	//Mirror y or x according to detected mirroring
	if (this->m_mirrored == 1)
	{
		y = m_texture->m_height - y;
	}
	else if (this->m_mirrored == 2)
	{
		x = m_texture->m_width - x;
	}

	double tmp_x = x;
	double tmp_y = y;

	if (!this->m_texture->m_isPattern)
	{
		//apply transformation
		tmp_x = m_transformationMatrix[0] * x + m_transformationMatrix[1] * y + m_transformationMatrix[2];
		tmp_y = m_transformationMatrix[3] * x + m_transformationMatrix[4] * y + m_transformationMatrix[5];
	}
}

}

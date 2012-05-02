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
 * Texture.tcc
 *
 *  @date 08.09.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

namespace lssr {

Texture::Texture()
{
	this->m_data   		= 0;
	this->m_textureClass 	= 0;

}

Texture::Texture(unsigned short int width, unsigned short int height, unsigned char numChannels,
		 unsigned char numBytesPerChan, unsigned short int textureClass)
{
	this->m_width 		= width;
	this->m_height 		= height;
	this->m_numChannels 	= numChannels;
	this->m_numBytesPerChan = numBytesPerChan;

	m_data = new char**[height];
	for (int y = 0; y < height; y++)
	{
		m_data[y] = new char*[width];
		for(int x = 0; x < width; x++)
		{
			m_data[y][x] = new char[numChannels * numBytesPerChan];
		}
	}

	this->m_textureClass	= textureClass;
}

Texture::~Texture() {
	for(int y = 0; y < m_height; y++)
	{
		for (int x = 0; x < m_width; x++)
		{
			delete m_data[y][x];
		}
		delete m_data[y];
	}
	delete m_data;
}

}

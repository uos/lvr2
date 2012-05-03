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
 * Texture.cpp
 *
 *  @date 08.09.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

namespace lssr {

float Texture::m_texelSize = 1;

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
	m_data = new char[width * height * numChannels * numBytesPerChan];
	this->m_textureClass	= textureClass;
}

Texture::Texture(Texture &other)
{
	this->m_width 		= other.m_width;
	this->m_height 		= other.m_height;
	this->m_numChannels 	= other.m_numChannels;
	this->m_numBytesPerChan = other.m_numBytesPerChan;
	m_data = new char[m_width * m_height * m_numChannels * m_numBytesPerChan];
	memcpy(m_data, other.m_data, m_width * m_height * m_numChannels * m_numBytesPerChan);
	this->m_textureClass	= other.m_textureClass;
}

Texture::~Texture() {
	delete m_data;
}

}

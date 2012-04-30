/* Copyright (C) 2012 Uni Osnabrück
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


 /**
 * @file       TextureIO.cpp
 * @brief      I/O support for texture package files (implementation).
 * @author     Kim Oliver Rinnewitz (krinnewitz), krinnewitz@uos.de
 * @date       Created:       2012-04-22 13:28:28
 */


#include "TextureIO.hpp"

#include <cstring>
#include <ctime>
#include <sstream>
#include <fstream>
#include "Timestamp.hpp"


namespace lssr
{


TextureIO::TextureIO(string filename)
{
	m_currentIndex = 0;
	m_filename = filename;
	ifstream in(m_filename.c_str());
	
	//test if file exists
	if()
	{	
		//file exists -> read all textures from the file
		
		//read number of textures
		size_t numTextures = 0;
		
		for (int i = 0; i < numTextures; i++)
		{
			
		}	
	}
	in.close();
}

size_t TextureIO::add(Texture* t)
{
	m_textures.add(t);	
	return m_textures.size() - 1;
}

void TextureIO::remove (size_t index)
{
	m_textures.erase(m_textures.begin() + index);
}

void TextureIO::update (size_t index, Texture* t)
{
	delete m_textures[i];  //TODO: copy instead of changing the pointer?
	m_textures[i] = t;
}

Texture* TextureIO::get(size_t index)
{
	return m_textures[index];
}

Texture* TextureIO::getNext()
{	
	if (m_currentIndex >= m_textures.size())
	{
		return 0;
	}
	else
	{
		return m_textures[m_curentIndex++];
	}
}

void TextureIO::write()
{
	ofstream out(m_filename.c_str());

	out<<m_textures.size();
	
	for (int i = 0; i < m_textures.size(); i++)
	{
		out 	<< m_textures[i]->m_textureClass 
			<< m_textures[i]->m_width 
			<< m_textures[i]->m_height 
			<< m_textures[i]->m_numChannels
			<< m_textures[i]->m_numBytesPerChan;
		out	<< m_textures[i]->m_data;
	}
	
	out.close();

}


}//namespace lssr

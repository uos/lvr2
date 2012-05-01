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
	m_currentIndex 	= 0;
	m_filename 	= filename;

	ifstream in(m_filename.c_str(), ios::in|ios::binary);
	
	if(in.good())
	{	
		//read all textures from the file
		
		//buffers for system independent I/O
		uint16_t ui16buf;
		uint8_t  ui8buf;

		//read number of textures: 2 Bytes
		size_t numTextures = 0;
		in.read(&numTextures, 2);
	
		//read all textures from the file	
		for (int i = 0; i < numTextures; i++)
		{
			Texture* t = new Texture();
		
			//read texture class: 2 Bytes
			in.read(&ui16buf, 2);	
			t->m_textureClass = ui16buf;
			
			//read texture width: 2 Bytes
			in.read(&ui16buf, 2);	
			t->width = ui16buf;
			
			//read texture height: 2 Bytes
			in.read(&ui16buf, 2);	
			t->height = ui16buf;

			//read number of channels and number of bytes per channel: 1 Byte
			in.read(&ui8buf, 1);	
			t->m_numChannels = (ui8buf & 0xf0) >> 4;
			t->m_numBytesPerChan = ui8buf & 0x0f;
			
			//allocate memory for the image data
			t->m_data = malloc(t->m_width * t->m_height * t->m_numChannels * t->m_numBytesPerChan);			

			//read image data line by line
			for (size_t y = 0; y < t->m_height; y++)
			{
				in.read(t->m_data[y], t->m_width * t->m_numChannels * t->m_numBytesPerChan);
			}
	

			m_textures.push_back(t);
		}	
	}
	in.close();
}

size_t TextureIO::add(Texture* t)
{
	m_textures.push_back(t);	
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
	ofstream out(m_filename.c_str(), ios::out|ios::binary);

	//buffers for system independent I/O
	uint16_t ui16buf;
	uint8_t  ui8buf;

	//Write number of textures in package: 2 Bytes
	ui16buf = m_textures.sze();
	out.write(&ui16buf, 2);

	//write all textures to the file	
	for (int i = 0; i < m_textures.size(); i++)
	{
		//write texture class: 2 Bytes
	        ui16buf = m_textures[i]->m_textureClass;	
		out.write(&ui16buf, 2);
		
		//write texture width: 2 Bytes
		ui16buf = m_textures[i]->m_width;
		out.write(&ui16buf, 2);

		//write texture height: 2 Bytes
		ui16buf = m_textures[i]->m_height;
		out.write(&ui16buf, 2);

		//write number of channels and number of bytes per channel 1 Byte
		ui8buf = (m_textures[i]->m_numChannels << 4) | m_textures[i]->m_numBytesPerChan;
		out.write(&ui8buf, 1);

		//write image data line by line
		for (int y = 0; y < m_textures[i]->m_height; y++)
		{
			out.write(m_textures[i]->m_data[y], m_textures[i]->m_width * m_textures[i]->m_numChannels * m_textures[i]->m_numBytesPerChan);
		}
	
	}
	
	out.close();

}


}//namespace lssr

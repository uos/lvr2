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



namespace lssr
{


TextureIO::TextureIO(string filename)
{
	m_filename 	= filename;

	std::ifstream in(m_filename.c_str(), std::ios::in|std::ios::binary);
	
	if(in.good())
	{	
		//read all textures from the file
		
		//buffers for system independent I/O
		uint16_t ui16buf;
		uint8_t  ui8buf;

		//read number of textures: 2 Bytes
		size_t numTextures = 0;
		in.read((char*)&numTextures, 2);
	
		//read all textures from the file	
		for (int i = 0; i < numTextures; i++)
		{
			Texture* t = new Texture();
		
			//read texture class: 2 Bytes
			in.read((char*)&ui16buf, 2);	
			t->m_textureClass = ui16buf;
			
			//read texture width: 2 Bytes
			in.read((char*)&ui16buf, 2);	
			t->m_width = ui16buf;
			
			//read texture height: 2 Bytes
			in.read((char*)&ui16buf, 2);	
			t->m_height = ui16buf;

			//read number of channels, number of bytes per channel and whether this texture is a pattern: 1 Byte
			in.read((char*)&ui8buf, 1);	
			t->m_numChannels = (ui8buf & 0xf0) >> 4;
			t->m_numBytesPerChan = (ui8buf & 0x0e) >> 1;
			t->m_isPattern = ui8buf & 0x01 == 1;
			
			//allocate memory for the image data
			t->m_data = new unsigned char[t->m_width * t->m_height * t->m_numChannels * t->m_numBytesPerChan];

			//read image data
			in.read((char*)t->m_data, t->m_width * t->m_height *t->m_numChannels * t->m_numBytesPerChan);
	
			//read number of features: 2 Bytes
			in.read((char*)&ui16buf, 2);	
			t->m_numFeatures = ui16buf;

			//read number of components: 1 Byte
			in.read((char*)&ui8buf, 1);
			t->m_numFeatureComponents = ui8buf;	

			//allocate memory for the feature descriptors
			t->m_featureDescriptors = new float[t->m_numFeatures * t->m_numFeatureComponents];
			//read feature descriptors
			in.read((char*)t->m_featureDescriptors, t->m_numFeatures * t->m_numFeatureComponents * sizeof(float));

			//allocate memory for the feature positions
			t->m_keyPoints = new float[t->m_numFeatures * 2];
			//read feature positions
			in.read((char*)t->m_keyPoints, t->m_numFeatures * 2 * sizeof(float));
			
			//read statistics
			t->m_stats = new float[14];
			in.read((char*)t->m_stats, 14 * sizeof(float));

			//read number of CCV colors: 1 Byte
			in.read((char*)&ui8buf, 1);
			t->m_numCCVColors = ui8buf;
		
			//read CCV
			t->m_CCV = new unsigned long[t->m_numCCVColors * 2 * 3];
			in.read((char*)t->m_CCV, t->m_numCCVColors * 2 * 3 * sizeof(unsigned long));

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
	delete m_textures[index]; 
	m_textures[index] = t;
}

void TextureIO::write()
{ 
	std::ofstream out(m_filename.c_str(), std::ios::out|std::ios::binary);

	//buffers for system independent I/O
	uint16_t ui16buf;
	uint8_t  ui8buf;

	//Write number of textures in package: 2 Bytes
	ui16buf = m_textures.size();
	out.write((char*)&ui16buf, 2);

	//write all textures to the file	
	for (int i = 0; i < m_textures.size(); i++)
	{
		//write texture class: 2 Bytes
	        ui16buf = m_textures[i]->m_textureClass;	
		out.write((char*)&ui16buf, 2);
		
		//write texture width: 2 Bytes
		ui16buf = m_textures[i]->m_width;
		out.write((char*)&ui16buf, 2);

		//write texture height: 2 Bytes
		ui16buf = m_textures[i]->m_height;
		out.write((char*)&ui16buf, 2);

		//write number of channels, number of bytes per channel and whether pattern or not: 1 Byte
		ui8buf = (m_textures[i]->m_numChannels << 4) | (m_textures[i]->m_numBytesPerChan << 1) | (m_textures[i]->m_isPattern ? 0x01 : 0x00);
		out.write((char*)&ui8buf, 1);

		//write image data
		out.write((char*)m_textures[i]->m_data, m_textures[i]->m_width *  m_textures[i]->m_height * 
							m_textures[i]->m_numChannels *  m_textures[i]->m_numBytesPerChan);

		//write number of features: 2 Bytes
		ui16buf = m_textures[i]->m_numFeatures;
		out.write((char*)&ui16buf, 2);

		//write number of components per feature descriptor: 1 Byte
		ui8buf =  m_textures[i]->m_numFeatureComponents;
		out.write((char*)&ui8buf, 1);

		//write feature descriptors
		out.write((char*)m_textures[i]->m_featureDescriptors, m_textures[i]->m_numFeatures *  m_textures[i]->m_numFeatureComponents * sizeof(float));

		//write feature positions
		out.write((char*)m_textures[i]->m_keyPoints, m_textures[i]->m_numFeatures *  2 * sizeof(float));

		//write statistical values
		out.write((char*)m_textures[i]->m_stats, 14 * sizeof(float));

		//write number of CCV colors: 1 Byte
		ui8buf = m_textures[i]->m_numCCVColors;
		out.write((char*)&ui8buf, 1);

		//write CCV
		out.write((char*)m_textures[i]->m_CCV, m_textures[i]->m_numCCVColors * 2 * sizeof(unsigned long) * 3);

	}
	
	out.close();

}


}//namespace lssr

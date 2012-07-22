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

#include "Texture.hpp"

namespace lssr {

float Texture::m_texelSize = 1;

Texture::Texture()
{
	this->m_data   		   	= 0;
	this->m_textureClass 	   	= 0;
	this->m_featureDescriptors 	= 0;
	this->m_width		   	= 0;
	this->m_height		   	= 0;
	this->m_numChannels	   	= 0;
	this->m_numBytesPerChan	   	= 0;
	this->m_numFeatures	   	= 0;
	this->m_numFeatureComponents	= 0;
	this->m_stats			= 0;
	this->m_isPattern		= false;
	this->m_numCCVColors		= 0;
	this->m_CCV			= 0;
}

Texture::Texture(unsigned short int width, unsigned short int height, unsigned char numChannels,
		 unsigned char numBytesPerChan, unsigned short int textureClass, unsigned short int numFeatures,
		 unsigned char numFeatureComponents, float* features, float* stats, bool isPattern, 
		 unsigned char numCCVColors, unsigned long* CCV)
{
	this->m_width 			= width;
	this->m_height 			= height;
	this->m_numChannels 		= numChannels;
	this->m_numBytesPerChan 	= numBytesPerChan;
	m_data = new unsigned char[width * height * numChannels * numBytesPerChan];
	this->m_textureClass		= textureClass;
	this->m_numFeatures	   	= numFeatures;
	this->m_numFeatureComponents	= numFeatureComponents;
	this->m_featureDescriptors 	= features; 
	this->m_stats			= stats;
	this->m_isPattern		= isPattern;
	this->m_numCCVColors 		= numCCVColors;
	this->m_CCV			= CCV;
}

Texture::Texture(Texture &other)
{
	this->m_width 			= other.m_width;
	this->m_height 			= other.m_height;
	this->m_numChannels 		= other.m_numChannels;
	this->m_numBytesPerChan 	= other.m_numBytesPerChan;
	m_data = new unsigned char[m_width * m_height * m_numChannels * m_numBytesPerChan];
	memcpy(m_data, other.m_data, m_width * m_height * m_numChannels * m_numBytesPerChan);
	this->m_textureClass		= other.m_textureClass;
	this->m_numFeatures 		= other.m_numFeatures;
	this->m_numFeatureComponents	= other.m_numFeatureComponents;
	this->m_featureDescriptors 	= new float[m_numFeatures * m_numFeatureComponents];
	memcpy(m_featureDescriptors, other.m_featureDescriptors, m_numFeatures * m_numFeatureComponents * sizeof(float));
	this->m_stats 			= new float[14];
	memcpy(m_stats, other.m_stats, 14 * sizeof(float));
	this->m_isPattern		= other.m_isPattern;
	this->m_numCCVColors		= other.m_numCCVColors;
	this->m_CCV			= new unsigned long[3 * m_numCCVColors * 2];
	memcpy(m_CCV, other.m_CCV, 3 * m_numCCVColors * 2 * sizeof(unsigned long));
}

void Texture::save(int i)
{
	//round the texture size up to a size to base 2
	int sizeX = std::max(8.0, pow(2, ceil(log(m_width) / log(2))));
	int sizeY = std::max(8.0, pow(2, ceil(log(m_height) / log(2))));
	cv::Mat img(cv::Size(sizeX, sizeY), CV_MAKETYPE(m_numBytesPerChan * 8, m_numChannels), cv::Scalar(255,0,0));
	cv::Rect roi(0,sizeY-m_height, m_width, m_height);
	cv::Mat t = img(roi);
	cv::Mat data(cv::Size(m_width, m_height), CV_MAKETYPE(m_numBytesPerChan * 8, m_numChannels), m_data);
	data.copyTo(t);
	
	char fn[255];
	sprintf(fn, "texture_%d.ppm", i);
	cv::imwrite(fn, img);
}

Texture::~Texture() {
	delete m_data;
	delete m_featureDescriptors;
	delete m_stats;
	delete m_CCV;
}

}

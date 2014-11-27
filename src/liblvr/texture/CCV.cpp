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
 * CCV.cpp
 *
 *  @date 17.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "texture/CCV.hpp"
//#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

using namespace std;

namespace lvr {

CCV::CCV(Texture* t, int numColors, int coherenceThreshold)
{
	this->m_numColors	 	= numColors;
	this->m_coherenceThreshold 	= coherenceThreshold;
	this->m_numPix 			= t->m_width * t->m_height;

	//convert texture to cv::Mat
	cv::Mat img(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);

	//split the image into its' r, g and b channel	
	cv::Mat img_planes[3];
	cv::split(img, img_planes);	
	
	//calculate the CCVs
	m_CCV_r = calculateCCV(img_planes[0]);
	m_CCV_g = calculateCCV(img_planes[1]);
	m_CCV_b = calculateCCV(img_planes[2]);
	
	
}

CCV::CCV(const cv::Mat &t, int numColors, int coherenceThreshold)
{
	this->m_numColors 		= numColors;
	this->m_coherenceThreshold 	= coherenceThreshold;
	this->m_numPix 			= t.rows * t.cols;
	
	//split the image into its' r, g and b channel	
	cv::Mat img_planes[3];
	cv::split(t, img_planes);	

	//calculate the CCVs
	m_CCV_r = calculateCCV(img_planes[0]);
	m_CCV_g = calculateCCV(img_planes[1]);
	m_CCV_b = calculateCCV(img_planes[2]);
}


CCV::CCV(Texture* t)
{
	this->m_numColors 		= t->m_numCCVColors;
	this->m_numPix 			= t->m_width * t->m_height;

	//set the CCVs
	m_CCV_r = fromArray(t->m_CCV);
	m_CCV_g = fromArray(&t->m_CCV[m_numColors * 2]);
	m_CCV_b = fromArray(&t->m_CCV[m_numColors * 2 * 2]);
}

std::map< unsigned char, std::pair<unsigned long, unsigned long> > CCV::fromArray(unsigned long int* arr)
{
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > ccv;
	for (int i = 0; i < 2 * m_numColors; i += 2)
	{	
		ccv[i / 2 + 0].first  = arr[i + 0];
		ccv[i / 2 + 1].second = arr[i + 1];
	}
	return ccv;

}

void CCV::toArray_r(unsigned long int* arr)
{
	for (int i = 0; i < 2 * m_numColors; i++)
	{	
		arr[i] = 0;
	}
	std::map< unsigned char, std::pair<unsigned long, unsigned long> >::iterator ccvit;
	//r
	for(ccvit = this->m_CCV_r.begin(); ccvit != this->m_CCV_r.end(); ccvit++)
	{
		arr[ccvit->first * 2 + 0] = ccvit->second.first;
		arr[ccvit->first * 2 + 1] = ccvit->second.second;
	}
}

void CCV::toArray_g(unsigned long int* arr)
{
	for (int i = 0; i < 2 * m_numColors; i++)
	{	
		arr[i] = 0;
	}
	std::map< unsigned char, std::pair<unsigned long, unsigned long> >::iterator ccvit;
	//g
	for(ccvit = this->m_CCV_g.begin(); ccvit != this->m_CCV_g.end(); ccvit++)
	{
		arr[ccvit->first * 2 + 0] = ccvit->second.first;
		arr[ccvit->first * 2 + 1] = ccvit->second.second;
	}
}

void CCV::toArray_b(unsigned long int* arr)
{
	for (int i = 0; i < 2 * m_numColors; i++)
	{	
		arr[i] = 0;
	}
	std::map< unsigned char, std::pair<unsigned long, unsigned long> >::iterator ccvit;
	//b
	for(ccvit = this->m_CCV_b.begin(); ccvit != this->m_CCV_b.end(); ccvit++)
	{
		arr[ccvit->first * 2 + 0] = ccvit->second.first;
		arr[ccvit->first * 2 + 1] = ccvit->second.second;
	}
}


std::map<unsigned short, std::pair<unsigned char, unsigned long> >CCV::calcCoherence(cv::Mat inputColors, cv::Mat inputLabels)
{
	//1 channel pointer to input image
	cv::Mat_<ushort>& ptrInputLabels = (cv::Mat_<ushort>&)inputLabels;
	//1 channel pointer to input colors image
	cv::Mat_<unsigned char>& ptrInputColors = (cv::Mat_<unsigned char>&)inputColors;

	//Map to hold the number of pixels and the color per label
	std::map<unsigned short, std::pair<unsigned char, unsigned long> > coherences;

	//calculate coherence values per label	
	for (int y = 0; y < inputLabels.size().height; y++)
	{
		for(int x = 0; x < inputLabels.size().width; x++)
		{
			if (coherences.find(ptrInputLabels(y,x)) != coherences.end())
			{
				coherences[ptrInputLabels(y,x)].second++;
			}
			else
			{
				coherences[ptrInputLabels(y,x)].second = 1;
				coherences[ptrInputLabels(y,x)].first = ptrInputColors(y,x);
			}
		}
	}

	return coherences;
}


std::map< unsigned char, std::pair<unsigned long, unsigned long> > CCV::calculateCCV(cv::Mat img)
{
	//blurred image
	cv::Mat blurred;

	//color reduced image
	cv::Mat reduced;

	//connected components
	cv::Mat labledComps;

	//Step 1: Blur the image slightly with a 3x3 box filter
	cv::blur(img, blurred, cv::Size(3,3)); //3x3 box filter

	//Step 2: Discretize the color space and reduce the number
	//	  colors to m_numColors
	ImageProcessor::reduceColorsG(blurred, reduced, m_numColors);

	//Step 3: Label connected components in the image in order
	//	  to determine the coherence of each pixel. The 
	//	  coherence is the size of the connected component
	//	  of the current pixel.
	ImageProcessor::connectedCompLabeling(reduced, labledComps);	
	//  label         color  size
	std::map<unsigned short, std::pair<unsigned char, unsigned long> > coherenceMap = calcCoherence(reduced, labledComps);	

	//Step 4: Calculate the CCV
	//This map holds the alpha and beta values for each color and can be referred to 
	//as the color coherence vector.
	//   color        alpha  beta

	std::map< unsigned char, std::pair<unsigned long, unsigned long> > ccv;

	//Iterator over the coherenceMap
	std::map<unsigned short, std::pair<unsigned char, unsigned long> >::iterator it;

	//Walk through the coherence map and sum up the incoherent and 
	//coherent pixels for every color
	for(it = coherenceMap.begin(); it != coherenceMap.end(); it++)
	{
		if (ccv.find(it->second.first) != ccv.end())
		{
			//we already have this color in the ccv
			if (it->second.second >= m_coherenceThreshold)
			{
				//pixels in current blob are coherent -> increase alpha
				ccv[it->second.first].first += it->second.second;
			}
			else
			{
				//pixels in current blob are incoherent -> increase beta
				ccv[it->second.first].second += it->second.second;
			}
		}
		else
		{
			//we don't have this color in our ccv yet and need to add it
			if (it->second.second >= m_coherenceThreshold)
			{
				//pixels in current blob are coherent -> set alpha
				ccv[it->second.first].first = it->second.second;
				ccv[it->second.first].second = 0;
			}
			else
			{
				//pixels in current blob are incoherent -> set beta
				ccv[it->second.first].first = 0;
				ccv[it->second.first].second = it->second.second;
			}
		}
			
	}

	//set unused colors to 0	
	for(unsigned char c = 0; c < m_numColors; c++)
	{
		if (ccv.find(c) == ccv.end())
		{
			ccv[c].first  = 0;
			ccv[c].second = 0;
		}
	}

	return ccv;
}


float CCV::compareTo(CCV* other)
{
	float result = 0;

	std::map< unsigned char, std::pair<unsigned long, unsigned long> >::iterator ccvit;

	//r
	for(ccvit = this->m_CCV_r.begin(); ccvit != this->m_CCV_r.end(); ccvit++)
	{
		//|alpha1 - alpha2| + |beta1 - beta2|
		result += fabs((int)ccvit->second.first / (1.0f * this->m_numPix)  - (int)other->m_CCV_r[ccvit->first].first  / (1.0f * other->m_numPix))
			+ fabs((int)ccvit->second.second / (1.0f * this->m_numPix) - (int)other->m_CCV_r[ccvit->first].second / (1.0f * other->m_numPix));
	}
	//g
	for(ccvit = this->m_CCV_g.begin(); ccvit != this->m_CCV_g.end(); ccvit++)
	{
		//|alpha1 - alpha2| + |beta1 - beta2|
		result += fabs((int)ccvit->second.first / (1.0f * this->m_numPix)  - (int)other->m_CCV_g[ccvit->first].first  / (1.0f * other->m_numPix))
			+ fabs((int)ccvit->second.second / (1.0f * this->m_numPix) - (int)other->m_CCV_g[ccvit->first].second / (1.0f * other->m_numPix));
	}
	//b
	for(ccvit = this->m_CCV_b.begin(); ccvit != this->m_CCV_b.end(); ccvit++)
	{
		//|alpha1 - alpha2| + |beta1 - beta2|
		result += fabs((int)ccvit->second.first / (1.0f * this->m_numPix)  - (int)other->m_CCV_b[ccvit->first].first  / (1.0f * other->m_numPix))
			+ fabs((int)ccvit->second.second / (1.0f * this->m_numPix) - (int)other->m_CCV_b[ccvit->first].second / (1.0f * other->m_numPix));
	}

	return result;
}

}

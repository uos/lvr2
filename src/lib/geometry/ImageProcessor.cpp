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
 * ImageProcessor.cpp
 *
 *  @date 12.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "ImageProcessor.hpp"
#include <opencv/cv.h>

namespace lssr {

void ImageProcessor::calcSURF(Texture* tex)
{
	cv::Mat img1(cv::Size(tex->m_width, tex->m_height), CV_MAKETYPE(tex->m_numBytesPerChan * 8,
			tex->m_numChannels), tex->m_data);
	//convert image to gray scale
	cv::cvtColor(img1, img1, CV_RGB2GRAY);
	
	//initialize SURF objects
	cv::SurfFeatureDetector detector(100);
	cv::SurfDescriptorExtractor extractor;

	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;

	//calculate SURF features for the image
	detector.detect( img1, keyPoints );
	extractor.compute( img1, keyPoints, descriptors );

	//return the results
	tex->m_numFeatures 		= descriptors.rows;
	tex->m_numFeatureComponents	= descriptors.cols;
	tex->m_featureDescriptors = new float[descriptors.rows * descriptors.cols];
	for (int r = 0; r < descriptors.rows; r++)
	{
		for (int c = 0; c < descriptors.cols; c++)
		{
			tex->m_featureDescriptors[r * descriptors.cols + c] = descriptors.at<float>(r, c);
		}
	}
}

float ImageProcessor::compareTexturesSURF(Texture* tex1, Texture* tex2)
{
	float result = FLT_MAX;

	//convert float arrays to cv::Mat
	cv::Mat descriptors1(tex1->m_numFeatures, tex1->m_numFeatureComponents, CV_32FC1);
	cv::Mat descriptors2(tex2->m_numFeatures, tex2->m_numFeatureComponents, CV_32FC1);
	
	if(tex1->m_numFeatures != 0 && tex2->m_numFeatures != 0)
	{
		
		result = 0;

		//calculate matching
		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		matcher.match( descriptors1, descriptors2, matches);

		//search best match
		double minDist = 100;
		for (int i = 0; i < matches.size(); i++)
		{ 
			if(matches[i].distance < minDist) minDist = matches[i].distance;
		}

		//Calculate result. Only good matches are considered.
		int numGoodMatches = 0;
		for( int i = 0; i < matches.size(); i++ )
		{ 
			if(matches[i].distance <= 2 * minDist)
			{
				result += matches[i].distance * matches[i].distance;
				numGoodMatches++;
			}
		}
		result /= numGoodMatches;
	}
	return result;

}
}

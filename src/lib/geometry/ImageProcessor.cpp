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

namespace lssr {

float* ImageProcessor::calcSURF(cv::Mat &img, unsigned short &numFeatures, unsigned char &numComps)
{
	//convert image to gray scale
	cv::Mat img1;
	cv::cvtColor(img, img1, CV_RGB2GRAY);
	
	//initialize SURF objects
	cv::SurfFeatureDetector detector(1000); //TODO: calculate ht
	cv::SurfDescriptorExtractor extractor;

	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;

	//calculate SURF features for the image
	detector.detect( img1, keyPoints );
	extractor.compute( img1, keyPoints, descriptors );

	//return the results
	numFeatures 	= descriptors.rows;
	numComps	= descriptors.cols;
	float* result = new float[numFeatures * numComps];
	for (int r = 0; r < numFeatures; r++)
	{
		for (int c = 0; c < numComps; c++)
		{
			result[r * numComps + c] = descriptors.at<float>(r, c);
		}
	}
	return result;
}

}

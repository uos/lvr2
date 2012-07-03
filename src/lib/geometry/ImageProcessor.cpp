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

void ImageProcessor::reduceColors(cv::Mat input, cv::Mat &output, int numColors)
{
	//allocate output
	output = cv::Mat(input.size(), CV_8U);
	//3 channel pointer to input image
	cv::Mat_<cv::Vec3b>& ptrInput = (cv::Mat_<cv::Vec3b>&)input; 
	//1 channel pointer to output image
	cv::Mat_<uchar>& ptrOutput = (cv::Mat_<uchar>&)output;

	for (int y = 0; y < input.size().height; y++)
	{
		for(int x = 0; x < input.size().width; x++)
		{
			unsigned long int currCol = 0;
			currCol |= (ptrInput(y, x)[0]) << 16;
			currCol |= (ptrInput(y, x)[1]) <<  8;
			currCol |= (ptrInput(y, x)[2]) <<  0;
			ptrOutput(y,x) = currCol / (pow(2, 24) / numColors);
		}
	}
}

void ImageProcessor::calcSURF(Texture* tex)
{
	//convert texture to cv::Mat
	cv::Mat img1(cv::Size(tex->m_width, tex->m_height), CV_MAKETYPE(tex->m_numBytesPerChan * 8, tex->m_numChannels), tex->m_data);
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
	for (int r = 0; r < descriptors1.rows; r++)
	{
		for (int c = 0; c < descriptors1.cols; c++)
		{
			descriptors1.at<float>(r, c) = tex1->m_featureDescriptors[r * descriptors1.cols + c];
		}
	}
	cv::Mat descriptors2(tex2->m_numFeatures, tex2->m_numFeatureComponents, CV_32FC1);
	for (int r = 0; r < descriptors2.rows; r++)
	{
		for (int c = 0; c < descriptors2.cols; c++)
		{
			descriptors2.at<float>(r, c) = tex2->m_featureDescriptors[r * descriptors2.cols + c];
		}
	}
	
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

void ImageProcessor::autocorrDFT(const cv::Mat &img, cv::Mat &dst)
{
	//Convert image from unsigned char to float matrix
	cv::Mat fImg;
	img.convertTo(fImg, CV_32FC1);
	//Subtract the mean
	cv::Mat mean(fImg.size(), fImg.type(), cv::mean(fImg));
	cv::subtract(fImg, mean, fImg);
	
	//Calculate the optimal size for the dft output.
	//This increases speed.
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(2 * img.cols +1 );
	dftSize.height = cv::getOptimalDFTSize(2 * img.rows +1);
	
	//prepare the destination for the dft
	dst = cv::Mat(dftSize, CV_32FC1, cv::Scalar::all(0));
	
	//transform the image into the frequency domain
	cv::dft(fImg, dst);
	//calculate DST * DST (don't mind the fourth parameter. It is ignored)
	cv::mulSpectrums(dst, dst, dst, cv::DFT_INVERSE, true);
	//transform the result back to the image domain 
	cv::dft(dst, dst, cv::DFT_INVERSE | cv::DFT_SCALE);

	//norm the result
	cv::multiply(fImg,fImg,fImg);
	float denom = cv::sum(fImg)[0];
	dst = dst * (1/denom);

}


double ImageProcessor::getMinimalPattern(const cv::Mat &input, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize)
{
	const float epsilon = 0.00005;

	//make input gray scale 
	cv::Mat img;	
	cv::cvtColor(input, img, CV_RGB2GRAY);

	//calculate auto correlation
	cv::Mat ac;
	ImageProcessor::autocorrDFT(img, ac);
	cv::Mat_<float>& ptrAc = (cv::Mat_<float>&)ac;

	//search minimal pattern i.e. search the highest correlation in x and y direction
	sizeX = 0;
	sizeY = 0;

	//y direction
	for (int y = minimalPatternSize; y < ac.size().height / 2; y++)
	{
		for(int x = 1; x < ac.cols / 2; x++)
		{
			if (ptrAc(y, x) > ptrAc(sizeY, sizeX) + epsilon || sizeY == 0)
			{
				sizeY = y;	
				sizeX = x;
			}
		}
	}
/*
	sizeX = 0;
	
	//x direction
	for (int x = minimalPatternSize; x < ac.size().width / 2; x++)
	{
		if (ptrAc(sizeY, x) > ptrAc(sizeY, sizeX) + epsilon || sizeX == 0)
		{
			sizeX = x;	
		}
	}*/
	
	return ptrAc(sizeY, sizeX);
} 

float ImageProcessor::extractPattern(Texture* tex, Texture** dst)
{
	//convert texture to cv::Mat
	cv::Mat src(cv::Size(tex->m_width, tex->m_height), CV_MAKETYPE(tex->m_numBytesPerChan * 8, tex->m_numChannels), tex->m_data);
	//convert image to gray scale
	cv::cvtColor(src, src, CV_RGB2GRAY);

	//try to extract pattern
	unsigned int sizeX, sizeY;
	float result = ImageProcessor::getMinimalPattern(src, sizeX, sizeY, 10); //TODO Param
	
	//save the pattern
	cv::Mat pattern = cv::Mat(src, cv::Rect(0, 0, sizeX, sizeY));
	
	//convert the pattern to Texture
	*dst = new Texture(pattern.size().width, pattern.size().height, pattern.channels(), tex->m_numBytesPerChan, tex->m_textureClass, 0, 0 ,0, 0);
	memcpy((*dst)->m_data, pattern.data, pattern.size().width * pattern.size().height * pattern.channels() * tex->m_numBytesPerChan);

	return result;
}

void ImageProcessor::calcStats(Texture* t)
{
	//TODO
}

}

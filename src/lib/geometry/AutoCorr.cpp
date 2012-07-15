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
 * AutoCorr.cpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include "AutoCorr.hpp"

namespace lssr {
AutoCorr::AutoCorr(Texture *t)
{
	//convert texture to cv::Mat
	cv::Mat img1(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);

	//make input gray scale 
	cv::Mat img;	
	cv::cvtColor(img1, img, CV_RGB2GRAY);

	autocorrDFT(img, m_autocorr);
}

AutoCorr::AutoCorr(const cv::Mat &t)
{
	autocorrDFT(t, m_autocorr);
}

void AutoCorr::autocorrDFT(const cv::Mat &img, cv::Mat &dst)
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

void AutoCorr::getACX(const cv::Mat &ac, float* &output)
{
	//Allocate output
	output = new float[ac.cols];
	for(int x = 0; x < ac.cols; x++)
	{
		float rho_x = 0;
		for(int y = 0; y < ac.rows; y++)
		{
			rho_x += ac.at<float>(y,x);
		}
		output[x] = rho_x;
	}
}

void AutoCorr::getACY(const cv::Mat &ac, float* &output)
{
	//Allocate output
	output = new float[ac.rows];
	for(int y = 0; y < ac.rows; y++)
	{
		float rho_y = 0;
		for(int x = 0; x < ac.cols; x++)
		{
			rho_y += ac.at<float>(y,x);
		}
		output[y] = rho_y;
	}
}

float AutoCorr::calcStdDev(const int* data, int len)
{
	float result = 0;
	float mean = 0;
	for (int i = 0; i < len; i++)
	{
		mean += data[i];
	}	
	mean /= len;

	for (int i = 0; i < len; i++)
	{
		result += (data[i] - mean) * (data[i] - mean);
	}
	result /= len;
	result = sqrt(result);

	return result;
}

int AutoCorr::countPeaks(const float* data, float &stdDev, int len)
{
	const float epsilon = 0.0001;
	int result = 0;

	if (len < 2)
	{
		return 0;
	}

	int lastPeak = -1;

	bool curr_up = true;

	//Count boarders, too
	if (data[0] > data[1])
	{
		result++;
		lastPeak = 0;
		curr_up = false;
	}

	int* distances = new int[len];

	//Search for peaks
	for (int i = 1; i < len - 1; i++)
	{
		bool next_up = curr_up;
//		if (data[i] > data[i-1])  
		if (data[i] - data[i-1] > epsilon)
		{
			next_up = true;
		}
//		if (data[i] < data[i-1])  
		if (data[i] - data[i-1] < -epsilon)  
		{
			next_up = false;
		}
		if (next_up == false && curr_up == true)
		{
			//peak detected
			if (lastPeak != -1)
			{
				distances[result-1] = lastPeak - i;
			}
			result++;
			lastPeak = i;
		}
		curr_up = next_up;
	}

	if (data[len-1] > data[len-2])
	{
		if (lastPeak != -1)
		{
			distances[result-1] = lastPeak - (len - 1);
		}
		result++;
	}

	stdDev = calcStdDev(distances, result - 1);

	return result;
}

double AutoCorr::getMinimalPattern(unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10)
{
	const float epsilon = 0.00005;

	cv::Mat_<float>& ptrAc = (cv::Mat_<float>&)m_autocorr;

//===========================
	float *rho_x = 0;
	float *rho_y = 0;	
	getACX(m_autocorr, rho_x);
	getACY(m_autocorr, rho_y);
	float stdDevX = 0;
	float stdDevY = 0;
	int peaksX = countPeaks(rho_x, stdDevX, m_autocorr.cols);
	int peaksY = countPeaks(rho_y, stdDevY, m_autocorr.rows);
	std::cout<<"Peaks x:"<<peaksX<<"\t\t StdDev rho_x: "<<stdDevX/(m_autocorr.cols / peaksX)<<std::endl;
	std::cout<<"Peaks y:"<<peaksY<<"\t\t StdDev rho_y: "<<stdDevY/(m_autocorr.rows / peaksY)<<std::endl;
//==========================

	//search minimal pattern i.e. search the highest correlation in x and y direction
	sizeX = 0;
	sizeY = 0;

	//y direction
/*	for (int y = minimalPatternSize; y < ac.size().height / 2; y++)
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

	sizeX = 0;
	
	//x direction
	for (int x = minimalPatternSize; x < ac.size().width / 2; x++)
	{
		if (ptrAc(sizeY, x) > ptrAc(sizeY, sizeX) + epsilon || sizeX == 0)
		{
			sizeX = x;	
		}
	}
*/	
	sizeX = 1; sizeY = 1; //TODO: remove
	return ptrAc(sizeY, sizeX);
} 

AutoCorr::~AutoCorr()
{
	//TODO?!
}
}

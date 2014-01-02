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
 * CrossCorr.cpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include "texture/CrossCorr.hpp"

namespace lvr {
CrossCorr::CrossCorr(Texture *t1, Texture* t2)
{
	//convert texture to cv::Mat
	cv::Mat img1(cv::Size(t1->m_width, t1->m_height), CV_MAKETYPE(t1->m_numBytesPerChan * 8, t1->m_numChannels), t1->m_data);

	//convert texture to cv::Mat
	cv::Mat img2(cv::Size(t2->m_width, t2->m_height), CV_MAKETYPE(t2->m_numBytesPerChan * 8, t2->m_numChannels), t2->m_data);

	//make input gray scale 
	cv::Mat img1g;	
	cv::cvtColor(img1, img1g, CV_RGB2GRAY);

	//make input gray scale 
	cv::Mat img2g;	
	cv::cvtColor(img2, img2g, CV_RGB2GRAY);

	//calculate cross correlation
	crosscorrDFT(img1g, img2g, m_crosscorr);
}

CrossCorr::CrossCorr(const cv::Mat &t1, const cv::Mat &t2)
{
	cv::Mat img1g;	
	cv::cvtColor(t1, img1g, CV_RGB2GRAY);

	//make input gray scale 
	cv::Mat img2g;	
	cv::cvtColor(t2, img2g, CV_RGB2GRAY);

	//calculate cross correlation
	crosscorrDFT(img1g, img2g, m_crosscorr);
}

void CrossCorr::crosscorrDFT(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst)
{
	//Convert image1 from unsigned char to float matrix
	cv::Mat A;
	img1.convertTo(A, CV_32FC1);
	//Subtract the mean
	cv::Mat meanA(A.size(), A.type(), cv::mean(A));
	cv::subtract(A, meanA, A);
	

	//Convert image2 from unsigned char to float matrix
	cv::Mat B;
	img2.convertTo(B, CV_32FC1);
	//Subtract the mean
	cv::Mat meanB(B.size(), B.type(), cv::mean(B));
	cv::subtract(B, meanB, B);

	// reallocate the output array if needed
	dst.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());

	// compute the size of DFT transform
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows - 1);

	// allocate temporary buffers and initialize them with 0’s
	cv::Mat tempA(dftSize, A.type(), cv::Scalar::all(0));
	cv::Mat tempB(dftSize, B.type(), cv::Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB, respectively
	cv::Mat roiA(tempA, cv::Rect(0,0,A.cols,A.rows));
	A.copyTo(roiA);
	cv::Mat roiB(tempB, cv::Rect(0,0,B.cols,B.rows));
	B.copyTo(roiB);

	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	cv::dft(tempA, tempA, 0, A.rows);
	cv::dft(tempB, tempB, 0, B.rows);

	//calculate DFT1 * DFT2 (don't mind the fourth parameter. It is ignored)
	cv::mulSpectrums(tempA, tempB, tempA, cv::DFT_INVERSE, true);

	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// we need only the first C.rows of them, and thus we
	// pass nonzeroRows == C.rows
	cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, dst.rows);

	// now copy the result back to C.
	tempA(cv::Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);

	//norm the result
	cv::multiply(A,A,A);
	cv::multiply(B,B,B);
	float denom = sqrt(cv::sum(A)[0]) * sqrt(cv::sum(B)[0]);	
	dst = dst * (1/denom);
}


double CrossCorr::getMax(unsigned int &resX, unsigned int &resY)
{
	const float epsilon = 0.00005;

	cv::Mat_<float>& ptrCC = (cv::Mat_<float>&)m_crosscorr;

	resX = 0;
	resY = 0;
	
	for (unsigned int x = 0; x < m_crosscorr.size().width; x++)
	{
		for (unsigned int y = 0; y < m_crosscorr.size().height; y++)
		{
			if (ptrCC(y, x) > ptrCC(resY, resX) + epsilon)
			{
				resY = y;	
				resX = x;
			}
		}
	}
	return ptrCC(resY, resX);
} 

float CrossCorr::at(unsigned int x, unsigned int y)
{
	//cout << m_crosscorr.rows << " " << m_crosscorr.cols << " " << x << " " << y << endl; 
if(x < m_crosscorr.cols && y < m_crosscorr.rows)
{
	return m_crosscorr.at<float>(y, x);
}
	else
{
	return 0;
}
}

void CrossCorr::getCCX(float* &output)
{
	//Allocate output
	output = new float[m_crosscorr.cols];
	for(int x = 0; x < m_crosscorr.cols; x++)
	{
		float rho_x = 0;
		for(int y = 0; y < m_crosscorr.rows; y++)
		{
			rho_x += m_crosscorr.at<float>(y,x);
		}
		output[x] = rho_x;
	}
}

void CrossCorr::getCCY(float* &output)
{
	//Allocate output
	output = new float[m_crosscorr.rows];
	for(int y = 0; y < m_crosscorr.rows; y++)
	{
		float rho_y = 0;
		for(int x = 0; x < m_crosscorr.cols; x++)
		{
			rho_y += m_crosscorr.at<float>(y,x);
		}
		output[y] = rho_y;
	}
}
}

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
 * Transform.cpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "Transform.hpp"
#include <opencv/highgui.h>
#include <opencv/cv.h>

namespace lssr {

//DEBUG
void showMatchings(cv::Mat t1, cv::Mat t2, std::vector<cv::KeyPoint> keyPoints1, std::vector<cv::KeyPoint> keyPoints2, std::vector< cv::DMatch > matches)
{
//	std::vector< cv::DMatch > goodMatches;
//	for (int i = 0; i < matches.size(); i++)
//	{
//		if (matches[i].distance < 0.1)
//		{
//			std::cout<<"POMMES PIMMES PIMMES " << matches[i].distance<<std::endl;
//			goodMatches.push_back(matches[i]);
//		}
//	}
	
	cv::Mat result;
	cv::drawMatches(t1, keyPoints1, t2, keyPoints2, matches, result);
	std::cout<<keyPoints1.size()<<" "<<keyPoints2.size()<<std::endl;
	
	cv::startWindowThread();
	
	//show the reference image
	cv::namedWindow("MATCHES", CV_WINDOW_AUTOSIZE);
	cv::imshow("MATCHES", result);
	cv::waitKey();

	cv::destroyAllWindows();
	
}

Transform::Transform(Texture *t1, Texture* t2)
{
	//convert textures to cv::Mat
	cv::Mat img1(cv::Size(t1->m_width, t1->m_height), CV_MAKETYPE(t1->m_numBytesPerChan * 8, t1->m_numChannels), t1->m_data);
	cv::Mat img2(cv::Size(t2->m_width, t2->m_height), CV_MAKETYPE(t2->m_numBytesPerChan * 8, t2->m_numChannels), t2->m_data);
	m_img1 = img1;
	m_img2 = img2;

	//make input gray scale 
	cv::Mat img1g;	
	cv::cvtColor(img1, img1g, CV_RGB2GRAY);
	cv::Mat img2g;	
	cv::cvtColor(img2, img2g, CV_RGB2GRAY);

	//calculate rotation, translation and scaling
	std::vector<cv::KeyPoint> keyPoints1;
	cv::Mat descriptors1;
	ImageProcessor::floatArrToSURF(t1, keyPoints1, descriptors1);
	std::vector<cv::KeyPoint> keyPoints2;
	cv::Mat descriptors2;
	ImageProcessor::floatArrToSURF(t2, keyPoints2, descriptors2);

	calcTransform(img1g, img2g, keyPoints1, keyPoints2, descriptors1, descriptors2);
}

Transform::Transform(const cv::Mat &t1, const cv::Mat &t2)
{
	m_img1 = t1;
	m_img2 = t2;

	//calculate surf features
//	cv::SurfFeatureDetector detector(1);
	cv::Ptr<cv::FeatureDetector> detector(new cv::DynamicAdaptedFeatureDetector (new cv::SurfAdjuster(), 100, 110, 10));
	cv::SurfDescriptorExtractor extractor;

	std::vector<cv::KeyPoint> keyPoints1;
	cv::Mat descriptors1;
	std::vector<cv::KeyPoint> keyPoints2;
	cv::Mat descriptors2;

	//calculate SURF features for the first image
	detector->detect( t1, keyPoints1 );
	extractor.compute( t1, keyPoints1, descriptors1 );

	//calculate SURF features for the second image
	detector->detect( t2, keyPoints2 );
	extractor.compute( t2, keyPoints2, descriptors2 );

	//calculate rotation, translation and scaling
	calcTransform(t1, t2, keyPoints1, keyPoints2, descriptors1, descriptors2);
}

void Transform::calcTransform(const cv::Mat &t1, const cv::Mat &t2, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, cv::Mat desc1, cv::Mat desc2)

{
	//we need at least three corresponding points!
	if (kp1.size() > 2 && kp2.size() > 2)
	{
		//calculate matching
		cv::BruteForceMatcher<cv::L2<float> > matcher;
//		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		matcher.match( desc1, desc2, matches);

		//search 3 best matches
		double minDist1 = FLT_MAX;
		double minDist2 = FLT_MAX;
		double minDist3 = FLT_MAX;
		int best1 = -1;
		int best2 = -1;
		int best3 = -1;
		showMatchings(t1, t2, kp1, kp2, matches); //TODO: remove
		for (int i = 0; i < matches.size(); i++)
		{ 
			if(matches[i].distance < minDist3)
			{
				if(matches[i].distance < minDist2)
				{
					if(matches[i].distance < minDist1)
					{
						minDist3 = minDist2;
						best3 = best2;
						minDist2 = minDist1;
						best2 = best1;
						minDist1 = matches[i].distance;
						best1 = i;
					}
					else
					{
						minDist3 = minDist2;
						best3 = best2;
						minDist2 = matches[i].distance;
						best2 = i;
					}
				}
				else
				{
					minDist3 = matches[i].distance;
					best3 = i;
				}
			}
		}

		cv::Point2f p1[3] = {kp1[matches[best1].queryIdx].pt, kp1[matches[best2].queryIdx].pt, kp1[matches[best3].queryIdx].pt};
		cv::Point2f p2[3] = {kp2[matches[best1].trainIdx].pt, kp2[matches[best2].trainIdx].pt, kp2[matches[best3].trainIdx].pt};

		//calculate rotation, translation and scaling
		m_trans = cv::getAffineTransform(p1, p2);
	}
	else
	{
		//Not enough corresponding points. Return identity matrix.
		m_trans = cv::Mat(2, 3, CV_64FC1);
		m_trans.at<double>(0,0) = 1;
		m_trans.at<double>(0,1) = 0;
		m_trans.at<double>(0,2) = 0;
		m_trans.at<double>(1,0) = 0;
		m_trans.at<double>(1,1) = 1;
		m_trans.at<double>(1,2) = 0;
	}
}


cv::Mat Transform::apply()
{
	cv::Mat result;

	//apply the inverse transformation to the second image/texture	
	cv::Mat inverse_trans;
	cv::invertAffineTransform(m_trans, inverse_trans);
	cv::warpAffine(m_img2, result, inverse_trans, m_img2.size());
	return result;
}

double* Transform::getTransArr()
{
	double* result = new double[6];
	for (int i = 0; i < 6; i++)
	{
		result[i] = m_trans.reshape(0,1).at<double>(0,i);
	}	
	return result;
}

Transform::~Transform()
{
	//TODO?!
}
}

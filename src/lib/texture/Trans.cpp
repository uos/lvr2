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
 * Trans.cpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "Trans.hpp"
#include <opencv/highgui.h>
#include <opencv/cv.h>

namespace lssr {
	Trans::Trans(cv::Point2f* p1, cv::Point2f* p2, int w1, int h1, int w2, int h2)
	{
		m_votes = 1;

		//define 4 tiles of the first picture
		cv::Rect rect11(0	, 0	, w1/2, h1/2);
		cv::Rect rect12(w1/2	, 0	, w1/2, h1/2);
		cv::Rect rect13(0	, h1/2	, w1/2, h1/2);
		cv::Rect rect14(w1/2	, h1/2	, w1/2, h1/2);
		//define 4 tiles of the second picture
		cv::Rect rect21(0	, 0	, w2/2, h2/2);
		cv::Rect rect22(w2/2	, 0	, w2/2, h2/2);
		cv::Rect rect23(0	, h2/2	, w2/2, h2/2);
		cv::Rect rect24(w2/2	, h2/2	, w2/2, h2/2);

		int mirr0 = 0;
		int mirr1 = 0;
		int mirr2 = 0;
		
		for (int i = 0; i < 3; i++)
		{
			if (p1[i].inside(rect11) &&  p2[i].inside(rect23) || p1[i].inside(rect12) && p2[i].inside(rect24))
			{
				//Check if not rotated
				//TODO: x oder y??
				if (abs(p1[0].x - p2[0].x) < 2 && abs(p1[1].x - p2[1].x) < 2 && abs(p1[2].x - p2[2].x) < 2)
				{
					mirr1++;
				}
				else
				{
					std::cout<<"Rotation detected (1)"<<std::endl;
				}
			}
			else if (p1[i].inside(rect11) &&  p2[i].inside(rect22) || p1[i].inside(rect13) && p2[i].inside(rect24))
			{
				//Check if not rotated
				//TODO: x oder y??
				if (abs(p1[0].y - p2[0].y) < 2 && abs(p1[1].y - p2[1].y) < 2 && abs(p1[2].y - p2[2].y) < 2)
				{
					mirr2++;
				}
				else
				{
					std::cout<<"Rotation detected (2)"<<std::endl;
				}
			}
			else
			{
				mirr0++;
				std::cout<<"No Rotation detected"<<std::endl;
			}
		}
		m_mirrored = std::max(mirr0, std::max(mirr1, mirr2)) == mirr0 ? 0 : std::max(mirr1, mirr2) == mirr1 ? 1 : 2;
		if (m_mirrored == 1)
		{
			//flip key points of second texture at horizontal axis
			p2[0].y = h2 - p2[0].y;
			p2[1].y = h2 - p2[1].y;
			p2[2].y = h2 - p2[2].y;
		}
		else if (m_mirrored == 2)
		{
			//flip key points of second texture at vertical axis
			p2[0].x = w2 - p2[0].x;
			p2[1].x = w2 - p2[1].x;
			p2[2].x = w2 - p2[2].x;
		}
		else
		{
			
		}
		m_trans = cv::getAffineTransform(p1, p2);
	}
	
	bool Trans::operator==(Trans other)
	{
		int epsilon = 10;
		bool result = true;
		//check what happens to some random points when applying the transformation
		for (int i = 0; i < 5; i++)
		{
			int x = rand() % 3000;
			int y = rand() % 3000;
			int x_transformed_by_this  = this->m_trans.at<double>(0,0) * x + this->m_trans.at<double>(0,1) * y + this->m_trans.at<double>(0,2);
			int y_transformed_by_this  = this->m_trans.at<double>(1,0) * x + this->m_trans.at<double>(1,1) * y + this->m_trans.at<double>(1,2);
			int x_transformed_by_other = other.m_trans.at<double>(0,0) * x + other.m_trans.at<double>(0,1) * y + other.m_trans.at<double>(0,2);
			int y_transformed_by_other = other.m_trans.at<double>(1,0) * x + other.m_trans.at<double>(1,1) * y + other.m_trans.at<double>(1,2);
			if (abs(x_transformed_by_this - x_transformed_by_other) > epsilon || abs(y_transformed_by_this - y_transformed_by_other) > epsilon)
			{
				result = false;
			}
		}	
		return result;
	}
}

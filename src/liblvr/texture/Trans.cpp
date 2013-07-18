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

#include "texture/Trans.hpp"
#include <opencv/highgui.h>
#include <opencv/cv.h>

namespace lvr {
	Trans::Trans(cv::Point2f* p1, cv::Point2f* p2, int w1, int h1, int w2, int h2)
	{

		float epsilon = 2;

		m_votes = 1;

		int mirr0 = 0;
		int mirr1 = 0;
		int mirr2 = 0;
		
		//save orders of x coordinates of key points
		bool relationsX1[3] = {p1[0].x > p1[1].x, p1[1].x > p1[2].x, p1[0].x > p1[2].x};
		bool relationsX2[3] = {p2[0].x > p2[1].x, p2[1].x > p2[2].x, p2[0].x > p2[2].x};

		//save orders of y coordinates of key points
		bool relationsY1[3] = {p1[0].y > p1[1].y, p1[1].y > p1[2].y, p1[0].y > p1[2].y};
		bool relationsY2[3] = {p2[0].y > p2[1].y, p2[1].y > p2[2].y, p2[0].y > p2[2].y};

		m_mirrored = 0;

		if (relationsX1[0] != relationsX2[0] && relationsX1[1] != relationsX2[1] && relationsX1[2] != relationsX2[2])
		{
			//Check if not rotated
			if (!(relationsY1[0] != relationsY2[0] && relationsY1[1] != relationsY2[1] && relationsY1[2] != relationsY2[2]))
		//	if (abs(p1[0].y - p2[0].y) < epsilon && abs(p1[1].y - p2[1].y) < epsilon && abs(p1[2].y - p2[2].y) < epsilon)
			{
				m_mirrored = 2;
			}
			else
			{
		//		std::cout<<std::endl<<std::endl<<"Rotation (2)"<<std::endl;
			}
		}

		if (relationsY1[0] != relationsY2[0] && relationsY1[1] != relationsY2[1] && relationsY1[2] != relationsY2[2])
		{
			//check if not rotated 
		//	if (abs(p1[0].x - p2[0].x) < epsilon && abs(p1[1].x - p2[1].x) < epsilon && abs(p1[2].x - p2[2].x) < epsilon)
			if (!(relationsX1[0] != relationsX2[0] && relationsX1[1] != relationsX2[1] && relationsX1[2] != relationsX2[2]))
			{
				m_mirrored = 1;
			}
			else
			{
		//		std::cout<<std::endl<<std::endl<<"Rotation (1)"<<std::endl;
			}
		}

		if (m_mirrored == 1)
		{
		//	std::cout<<std::endl<<std::endl<<"MIRRORING STATE: 1"<<std::endl;
			//flip key points of second texture at horizontal axis
			p2[0].y = h2 - p2[0].y;
			p2[1].y = h2 - p2[1].y;
			p2[2].y = h2 - p2[2].y;
		}
		else if (m_mirrored == 2)
		{
		//	std::cout<<std::endl<<std::endl<<"MIRRORING STATE: 2"<<std::endl;
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

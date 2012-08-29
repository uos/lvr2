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
 * Transform.hpp
 *
 *  @date 06.08.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef TRANSFORM_HPP_
#define TRANSFORM_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include <geometry/Texture.hpp>
#include <geometry/ImageProcessor.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace lssr {


/**
 * @brief	This class calculates transformations between images/textures.
 */
class Transform {
public:


	/**
	* \brief Constructor. Calculates the transformation between the given textures
	*
	* \param	t1		The first texture
	* \param	t2		The second texture
	*
	*/
	Transform(Texture* t1, Texture* t2);

	/**
	* \brief Constructor. Calculates the transformation between the given textures
	*
	* \param	t1		The first texture
	* \param	t2		The second texture
	*
	*/
	Transform(const cv::Mat &t1, const cv::Mat &t2);

	/**
	 * \brief Applies the transformation to the second texture/image
 	 *
	 * \return	The transformed texture/image
	 */
	cv::Mat apply();

	/**
	 * \brief Returns the transformation matrix as a float array
	 *
	 * \return The transformation matrix
	 */
	double* getTransArr();

	/**
	 * Destructor.
	 */
	virtual ~Transform();
	

private:
	class Trans
	{
	public:
		Trans(cv::Point2f* p1, cv::Point2f* p2, int w1, int h1, int w2, int h2)
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
					//TODO: Check if not rotated
					mirr1++;
				}
				else if (p1[i].inside(rect11) &&  p2[i].inside(rect22) || p1[i].inside(rect13) && p2[i].inside(rect24))
				{
					//TODO: Check if not rotated
					mirr2++;
				}
				else
				{
					mirr0++;
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
		
		bool operator==(Trans other)
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
		
		///The number of votes this transformation has
		int m_votes;
	
		///The transformation matrix	
		cv::Mat m_trans;

		///0 = not mirrored, 1 = mirrored at horizontal axis, 2 = mirrored at vertical axis
		unsigned char m_mirrored;
		
	};
	
	/**
	 * \brief calculates the rotation, translation and scaling between the two given images
	 *
	 * \param	t1	The first image
	 * \param	t2	The second image
	 *
	 */
	void calcTransform(const cv::Mat &t1, const cv::Mat &t2, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, cv::Mat desc1, cv::Mat desc2);
	
	///The first image
	cv::Mat m_img1;
	
	///The second image
	cv::Mat m_img2;

	///The transformation
	cv::Mat m_trans;
	

	///The rotation angle
	float m_alpha;

	///The translation in x direction
	int m_tx;

	///The translation in x direction
	int m_ty;

	///The scaling
	float m_s;
};

}

#endif /* TRANSFORM_HPP_ */

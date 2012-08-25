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
		Trans(cv::Point2f* p1, cv::Point2f* p2)
		{
			m_votes = 1;
			m_trans = cv::getAffineTransform(p1, p2);
			m_mirrored = false; //TODO: check if mirrored
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
		
		int m_votes;	
		cv::Mat m_trans;
		bool m_mirrored;
		
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

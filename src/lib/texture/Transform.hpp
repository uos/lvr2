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
#include <texture/Texture.hpp>
#include <texture/ImageProcessor.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <texture/Trans.hpp>

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
	virtual ~Transform(){};
	

private:
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
};

}

#endif /* TRANSFORM_HPP_ */

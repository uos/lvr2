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
 * CrossCorr.hpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef CROSSCORR_HPP_
#define CROSSCORR_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include "Texture.hpp"

namespace lvr {


/**
 * @brief	This class provides auto correlation functions for texture analysis..
 */
class CrossCorr {
public:


	/**
	* \brief Constructor. Calculates the cross correaltion of the given Textures.
	*
	* \param	t1		The first texture
	* \param	t2		The second texture
	*
	*/
	CrossCorr(Texture* t1, Texture* t2);

	/**
	* \brief Constructor. Calculates the cross correaltion of the given Textures.
	*
	* \param	t1		The first texture
	* \param	t2		The second texture
	*
	*/
	CrossCorr(const cv::Mat &t1, const cv::Mat &t2);

	/**
	 * \brief Searches the maximum value in the given cross correlation.
	 *
	 * \param input	The cross correlation of two images
	 * \param resX	The x coordinate of the maximum value
	 * \param resY	The y coordinate of the maximum value
	 *
	 * \return The maximum value.
	 */
	double getMax(unsigned int &resX, unsigned int &resY);

	/**
	 * \brief Returns the cross correlation for the given lag
	 *
	 * \param	x 	The x coordinate of the lag
	 * \param	y 	The y coordinate of the lag
	 *
	 * \return	The cross correlation for the given lag
	 */
	float at(unsigned int x, unsigned int y);

	/**
	 * Destructor.
	 */
	virtual ~CrossCorr(){};

private:
	/**
	 * \brief 	Implementation of the cross correlation function using fourier transformation.
	 *		This implementation is quite fast and may be used for productive jobs. Cross
	 *		correlation can be calculated by transforming the input images the frequency
	 *		domain (getting the fourier transformations of the images), multiplying the
	 *		first spectrum with the second spectrum  and
	 *		transforming the result back to the image domain. 
	 *
	 * \param 	img1	The first image. Must be one channel gray scale.
	 * \param 	img2	The second image. Must be one channel gray scale.
	 * \param	dst	The destination to store the correlation values in. The result is NOT
				normed.
	 */
	void crosscorrDFT(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst);

	/**
	 * \brief	Calculates the sum of all rows for ecch column of
	 *		the given crosscorrelation matrix. It will return
	 *		a float array of length cc.cols.
	 *
	 * \param	cc	The cross correlation matrix
	 * \param	output	The destination where results are stored
	 */
	void getCCX(float* &output);

	/**
	 * \brief	Calculates the sum of all columns for ecch row of
	 *		the given crosscorrelation matrix. It will return
	 *		a float array of length cc.rows.
	 *
	 * \param	cc	The cross correlation matrix
	 * \param	output	The destination where results are stored
	 */
	void getCCY(float* &output);

	cv::Mat m_crosscorr;
};

}

#endif /* CROSSCORR_HPP_ */

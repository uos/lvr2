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
 * AutoCorr.hpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef AUTOCORR_HPP_
#define AUTOCORR_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include "Texture.hpp"
#include "CrossCorr.hpp"

namespace lssr {


/**
 * @brief	This class provides auto correlation functions for texture analysis..
 */
class AutoCorr {
public:


	/**
	* \brief Constructor. Calculates the auto correaltion for the given Texture.
	*
	* \param	t		The texture
	*
	*/
	AutoCorr(Texture* t);

	/**
	* \brief Constructor. Calculates the auto correaltion for the given Texture.
	*
	* \param	t		The texture
	*
	*/
	AutoCorr(const cv::Mat &t);

	/**
	 * \brief	Tries to find a pattern in an Image using the auto correlation
	 *		function. The result can be interpreted as a rectangle at the
	 *		position (sX,sY) of the input image with the width of sizeX and the
	 * 		height of sizeY.
	 *
	 * \param	sX			The starting x position of the found pattern
	 * \param	sY			The starting y position of the found pattern
	 * \param	sizeX			The resulting x size of the found pattern
	 * \param	sizeY			The resulting y size of the found pattern
	 *
	 * \return	A confidence  indicating the degree of success in extracting a 
	 *		pattern from the given image
	 */
	double getMinimalPattern(unsigned int &sX, unsigned int &sY, unsigned int &sizeX, unsigned int &sizeY);
	
	/**
	 * Destructor.
	 */
	virtual ~AutoCorr(){};

private:
	/**
	 * \brief 	Implementation of the auto correlation function using fourier transformation.
	 *		This implementation is quite fast and may be used for productive jobs. Auto
	 *		correlation can be calculated by transforming the image img into the frequency
	 *		domain (getting the fourier transformation IMG of img), calculating
	 *		IMG * IMG  and transforming the result back to the image domain. 
	 *
	 * \param 	img	The image to calculate the auto correlation for. Must be one channel
	 *			gray scale.
	 * \param	dst	The destination to store the correlation values in. The result is normed.
	 */
	void autocorrDFT(const cv::Mat &img, cv::Mat &dst);

	/**
	 * \brief	Calculates the sum of all rows for each column of
	 *		the given autocorrelation matrix. It will return
	 *		a float array of length ac.cols.
	 *
	 * \param	ac	The auto correlation matrix
	 * \param	output	The destination where results are stored
	 */
	void getACX(const cv::Mat &ac, float* &output);

	/**
	 * \brief	Calculates the sum of all columns for each row of
	 *		the given autocorrelation matrix. It will return
	 *		a float array of length ac.rows.
	 *
	 * \param	ac	The auto correlation matrix
	 * \param	output	The destination where results are stored
	 */
	void getACY(const cv::Mat &ac, float* &output);

	/**
	* \brief 	Calculates the standard deviation of the given 
	*		data array
	*
	* \param	data	The data array
	* \param	len 	The length of the data array
	*
	* \return 	The standard deviation of the given data array
	*/
	float calcStdDev(const int* data, int len);

	/**
	 * \brief 	Counts the number of peaks in the given data Array and
	 * 		returns the standard deviation of the distances between
	 *		the peaks
	 * \param	data		The data array
	 * \param	stdDev		The standard deviation of the distances
	 *				between the peaks
	 * \param	len		The length of the array
	 * \param	peaks		Array to hold the positions of all peaks
	 *
	 * \return	The number of peaks in the data array
	 */
	int countPeaks(const float* data, float &stdDev, int len, int peaks[]);

	/**
	 * \brief Calculates the "relative" height of a peak in the given data
	 *
	 * \param	peak	The position of the peak
	 * \param	data	The data
	 * \param	len	The length of the data
 	 *
	 * \return The "relative" height of the given peak in the data
	 */
	float calcPeakHeight(int peak, float* data, int len);

	/// Auto correlation	
	cv::Mat m_autocorr;
	
	/// The input image
	cv::Mat m_image;
};

}

#endif /* AUTOCORR_HPP_ */

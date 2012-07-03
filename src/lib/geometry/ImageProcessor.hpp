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
 * ImageProcessor.hpp
 *
 *  @date 12.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include <cstring>
#include <cstdio>
#include <geometry/Texture.hpp>

namespace lssr {


/**
 * @brief	This class provides image processing functions for texture generation
 */
class ImageProcessor {
public:

/**
 * \brief Reduces the number of colors in the given image
 * 
 * \param input		The input image to reduce the colors in.
			This must be a 3 channel image with 8 bit
			per channel.
 * \param output 	The destination to store the result in.
			This will be an 8 bit one channel image.
 * \param numColors	The maximum number of colors in the 
 *			output image. Note, that this value must
 *			be less than or equal to 256 since the 
 *			output image has only one 8 bit channel.
 */
static void reduceColors(cv::Mat input, cv::Mat &output, int numColors);

/**
 * \brief 	Calculates the SURF features for the given texture
 *
 * \param	tex		The texture to calculate the feature for
 */
static void calcSURF( Texture* tex);

/**
 * \brief 	Compares the given textures wrt to their SURF descriptors
 *
 * \param	tex1	The first texture
 * \param	tex2	The second texture
 *
 * \return 	The distance between the textures
 */
static float compareTexturesSURF(Texture* tex1, Texture* tex2);

/**
 * \brief	Tries to extract a pattern from the given texture
 *
 * \param 	tex	The texture to extract a pattern from
 * \param 	dst	The destination to store the pattern
 *
 * \return	A value indicating how "good" the pattern is.
 *		The higher the value, the "better" the pattern.
 */
static float extractPattern(Texture* tex, Texture** dst);

/**
 * \brief 	Calculates 14 statistical values for the given texture
 *
 * \param	The texture to calculate the stats for
 */
static void calcStats(Texture* t);


private:

/**
 * \brief 	Implementation of the auto correlation function using fourier transformation.
 *		This implementation is quite fast and may be used for productive jobs. Auto
 *		correlation can be calculated by transforming the image img into the frequency
 *		domain (getting the fourier transformation IMG of img), calculating
 *		IMG * IMG and transforming the result back to the image domain. 
 *
 * \param 	img	The image to calculate the auto correlation for. Must be one channel
 *			gray scale.
 * \param	dst	The destination to store the correlation values in. The result is NOT
			normed.
 */
static void autocorrDFT(const cv::Mat &img, cv::Mat &dst);

/**
 * \brief	Tries to find a pattern in an Image using the auto correlation
 *		function. The result can be interpreted as a rectangle at the
 *		origin (0,0) of the input image with the width of sizeX and the
 * 		height of sizeY.
 *
 * \param	input			The image to find a pattern in. Has to be
					a three channel	color (RGB) image.
 * \param	sizeX			The resulting x size of the found pattern
 * \param	sizeY			The resulting y size of the found pattern
 * \param	minimalPatternSize	The minimum acceptable x and y size of a
 *					pattern 
 *
 * \return	A confidence between 0 and 1 indicating the degree of success in
 *		extracting a pattern from the given image
 */
static double getMinimalPattern(const cv::Mat &input, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10);

};
}

#endif /* IMAGEPROCESSOR_HPP_ */

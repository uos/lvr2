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
#include <math.h>
//#include <boost/pending/disjoint_sets.hpp>
#include <texture/Texture.hpp>
#include <texture/Statistics.hpp>
#include <texture/AutoCorr.hpp>
#include <texture/CrossCorr.hpp>
#include <texture/CCV.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace lvr {


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
	 * \brief Reduces the number of colors in the given gray scale image
	 * 
	 * \param input		The input image to reduce the colors in.
				This must be a 1 channel image with 8 bit
				per channel.
	 * \param output 	The destination to store the result in.
				This will be an 8 bit one channel image.
	 * \param numColors	The maximum number of colors in the 
	 *			output image. Note, that this value must
	 *			be less than or equal to 256 since the 
	 *			output image has only one 8 bit channel.
	 */
	static void reduceColorsG(cv::Mat input, cv::Mat &output, int numColors);

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
	 * \param 	tex			The texture to extract a pattern from
	 * \param 	dst			The destination to store the pattern
	 *
	 * \return	A value indicating how "good" the pattern is.
	 *		The higher the value, the "better" the pattern.
	 */
	static float extractPattern(Texture* tex, Texture** dst);

	/**
	 * \brief 	Calculates 14 statistical values for the given texture
	 *
	 * \param	t		The texture to calculate the stats for
	 * \param	numColors	The number of gray levels to use
	 */
	static void calcStats(Texture* t, int numColors);

	/**
	 * \brief 	Labels connected components in the given image.
	 *		This is an implementation of the algorithm of
	 *		Rosenfeld et al
	 * 
	 * \param	input	The image to label connected components in
	 * \param	output	The destination to hold the labels
	 */
	static void connectedCompLabeling(cv::Mat input, cv::Mat &output);

	/**
	 * \brief 	Calculates the CCV for the given texture.
	 * 
   	 * \param	t			The texture
	 * \param	numColors		The number of colors to use
	 * \param	coherenceThreshold	The coherence threshold to use
	 *
	 */
	static void calcCCV(Texture* t, int numColors, int coherenceThreshold);

	/**
	 * \brief 	Compares the given textures wrt to their CCVs
	 *
	 * \param	tex1	The first texture
	 * \param	tex2	The second texture
	 *
	 * \return 	The distance between the textures
	 */
	static float compareTexturesCCV(Texture* tex1, Texture* tex2);

	/**
	 * \brief 	Compares the given textures wrt to their histograms
	 *
	 * \param	tex1	The first texture
	 * \param	tex2	The second texture
	 *
	 * \return 	The distance between the textures
	 */
	static float compareTexturesHist(Texture* tex1, Texture* tex2);

	/**
	 * \brief 	Compares the given textures wrt to their statistics
	 *
	 * \param	tex1	The first texture
	 * \param	tex2	The second texture
	 *
	 * \return 	The distance between the textures
	 */
	static float compareTexturesStats(Texture* tex1, Texture* tex2);
	
	/**
	 * \brief 	Compares the given textures wrt to the cross correlation
	 *
	 * \param	tex1	The first texture
	 * \param	tex2	The second texture
	 *
	 * \return 	The distance between the textures
	 */
	static float compareTexturesCrossCorr(Texture* tex1, Texture* tex2);

	/**
	 * \brief	Converts the float values stored with each texture back to 
	 *		OpenCV usable data structures.
	 *
         * \param	t	The texture to convert the values from
	 * \param	kp	The destination vector to hold the key points
	 * \param	desc	The destination matrix to hold the feature descriptors
         */
	static void floatArrToSURF(Texture* t, std::vector<cv::KeyPoint> &kp, cv::Mat &desc);

	static void showTexture(Texture* t, string caption);
	static void showTexture(cv::Mat img, string caption);
private:

	/**
	* \brief 	Implementation of the find algorithm for disjoint sets.
	* 
	* \param 	x	The element to find
	* \param	parent	The disjoint set data structure to work on (tree)
	*
	* \return 	The number of the set which contains the given element
	*/
	static unsigned long int find(unsigned long int x, unsigned long int parent[]);

	/**
	* \brief	Implementation of the union algorithm for disjoint sets.
	*
	* \param	x	The first set for the two sets to unite 
	* \param	y	The second set for the two sets to unite 
	* \param	parent	The disjoint set data structure to work on (tree)
	*/
	static void unite(unsigned long int x, unsigned long int y, unsigned long int parent[]);

};
}

#endif /* IMAGEPROCESSOR_HPP_ */

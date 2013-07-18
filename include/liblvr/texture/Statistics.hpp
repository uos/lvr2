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
 * Statistics.hpp
 *
 *  @date 24.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef STATISTICS_HPP_
#define STATISTICS_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include "Texture.hpp"
#include "ImageProcessor.hpp"

namespace lvr {


/**
 * @brief	This class provides statistical methods for texture analysis..
 */
class Statistics {
public:


	/**
	* \brief Constructor. Calculates the cooccurrence matrix for the given Texture.
	*
	* \param	t		The texture
	* \param	numColors	The number of gray levels to use
	*
	*/
	Statistics(Texture* t, int numColors);

	/**
	* \brief Constructor. Calculates the cooccurrence matrix for the given Texture.
	*
	* \param	t		The texture
	* \param	numColors	The number of gray levels to use
	*
	*/
	Statistics(const cv::Mat &t, int numColors);

	/**
	* \brief	Calculates the distance of two texture vectors
	*
	* \param	v1	The first texture vector
	* \param	v2	The second texture vector
	*
	* \return	The distance of the given texture vectors
	*/
	static float textureVectorDistance(float* v1, float* v2);

	/**
	* \brief	Calculates the distance of two texture vectors
	*
	* \param	v1	The first texture vector
	* \param	v2	The second texture vector
	* \param	coeffs	The coefficients to use for comparison
	*
	* \return	The distance of the given texture vectors
	*/
	static float textureVectorDistance(float* v1, float* v2, float* coeffs);

	/**
	 * Destructor.
	 */
	virtual ~Statistics();

	/**
	 * \brief	Calculates the angular second moment of the texture
	 *
	 * \return 	The angular second moment
	 */
	float calcASM();

	/**
	 * \brief	Calculates the contrast of the texture
	 *
	 * \return 	The contrast of the texture
	 */
	float calcContrast();

	/**
	 * \brief	Calculates the correlation of the texture
	 *
	 * \return 	The correlation of the texture
	 */
	float calcCorrelation();

	/**
	 * \brief	Calculates the sum of squares of the texture
	 *
	 * \return 	The sum of squares of the texture
	 */
	float calcSumOfSquares();

	/**
	 * \brief	Calculates the inverse difference moment of the texture
	 *		represented by the given cooccurrence matrix
	 *
	 * \return 	The inverse difference moment of the texture
	 */
	float calcInverseDifference();

	/**
	 * \brief	Calculates the sum average of the texture
	 *
	 * \return 	The sum average of the texture
	 */
	float calcSumAvg();

	/**
	 * \brief	Calculates the sum entropy of the texture
	 *
	 * \return 	The sum entropy of the texture
	 */
	float calcSumEntropy();

	/**
	 * \brief	Calculates the sum variance of the texture
	 *
	 * \return 	The sum variance of the texture
	 */
	float calcSumVariance();

	/**
	 * \brief	Calculates the entropy of the texture
	 *
	 * \return 	The entropy of the texture
	 */
	float calcEntropy();

	/**
	 * \brief	Calculates the difference variance of the texture
	 *
	 * \return 	The difference variance of the texture
	 */
	float calcDifferenceVariance();

	/**
	 * \brief	Calculates the difference entropy of the texture
	 *
	 * \return 	The difference entropy of the texture
	 */
	float calcDifferenceEntropy();

	/**
	 * \brief	Calculates the information measures 1 of the texture
	 *
	 * \return 	The information measures 1 of the texture
	 */
	float calcInformationMeasures1();

	/**
	 * \brief	Calculates the information measures 2 of the texture
	 *
	 * \return 	The information measures 2 of the texture
	 */
	float calcInformationMeasures2();

	/**
	 * \brief	Calculates the maximal correlation coefficient of the texture
	 *
	 * \return 	The maximal correlation coefficient of the texture
	 */
	float calcMaxCorrelationCoefficient();

	///The 14 coefficients for texture comparison
	static float m_coeffs[14];

private:

	/**
         * \brief calculates all cooccurence matrizes
         *
         * \param	t	The texture image
         */
	void calcCooc(const cv::Mat &t);

	/**
	 * \brief	Returns the i-th entry of the magrginal probability matrix
	 *		of the given cooccurrence matrix
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	i		The entry to get
	 *
	 *
	 */
	float px(float** com, int i);

	/**
	 * \brief	Returns the j-th entry of the magrginal probability matrix
	 *		of the given cooccurrence matrix
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	j		The entry to get
	 *
	 *
	 */
	float py(float** com, int j);

	/**
	 * \brief	Calculates p_{x+y}(k)	
	 *		
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	k		k
	 *
	 * \return 	p_{x+y}(k)
	 */
	float pxplusy(float** com,  int k);

	/**
	 * \brief	Calculates p_{x-y}(k)	
	 *		
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	k		k
	 *
	 * \return 	p_{x-y}(k)
	 */
	float pxminusy(float** com, int k);

	/**
	 * \brief	Calculates the entropy of the given cooccurrence
         *		matrix
	 *
  	 * \pram	com	The cooccurence matrix
	 *
	 * \return 	The entropy of the texture
	 */
	float calcEntropy(float** com);

	/**
	 * \brief	Calculates the sum entropy of the given cooccurrence
         *		matrix
	 *
  	 * \pram	com	The cooccurence matrix
	 *
	 * \return 	The entropy of the texture
	 */
	float calcSumEntropy(float** com);

	//The number of rows and cols of the cooccurrence matrix
	int m_numColors;

	//cooccurrence matrix for 0 degrees direction
	float** m_cooc0;
	
	//cooccurrence matrix for 45 degrees direction
	float** m_cooc1;

	//cooccurrence matrix for 90 degrees direction
	float** m_cooc2;

	//cooccurrence matrix for 135 degrees direction
	float** m_cooc3;

	static float epsilon;
};

}

#endif /* STATISTICS_HPP_ */

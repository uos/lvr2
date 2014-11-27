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
 * CCV.hpp
 *
 *  @date 17.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef CCV_HPP_
#define CCV_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include <map>
#include "Texture.hpp"
#include "ImageProcessor.hpp"

namespace lvr {


/**
 * @brief	This class provides statistical methods for texture analysis..
 */
class CCV {
public:


	/**
	* \brief Constructor. Calculates the CCVs for the given Texture.
	*
	* \param	t			The texture
	* \param	numColors		The number of gray levels to use
        * \param	coherenceThreshold	The coherence threshold
	*
	*/
	CCV(Texture* t, int numColors, int coherenceThreshold);

	/**
	* \brief Constructor. Calculates the CCVs for the given Texture.
	*
	* \param	t		The texture
	* \param	numColors	The number of gray levels to use
        * \param	coherenceThreshold	The coherence threshold
	*
	*/
	CCV(const cv::Mat &t, int numColors, int coherenceThreshold);

	/**
	* \brief Constructor. Loads the CCVs from the given Texture.
	*
	* \param	t		The texture
	*
	*/
	CCV(Texture* t);

	/**
	 * \brief	Calculates the distance to the given CCV.
	 *
	 * \param	other	The other CCV
	 *
	 * \return	The distance between the two CCVs
	 */
	float compareTo(CCV* other);

	/**
  	 * \brief	Exports the CCV to an array (R)
	 * 
	 * \param	arr	The allocated destination array
	 *
	 */
	void toArray_r(unsigned long* arr);

	/**
  	 * \brief	Exports the CCV to an array (G)
	 * 
	 * \param	arr	The allocated destination array
	 *
	 */
	void toArray_g(unsigned long* arr);

	/**
  	 * \brief	Exports the CCV to an array (B)
	 * 
	 * \param	arr	The allocated destination array
	 *
	 */
	void toArray_b(unsigned long* arr);

	/**
  	 * \brief	Imports a CCV to an array
	 * 
	 * \param	arr	The CCV array
	 *
	 * \return	The imported CCV map
	 */
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > fromArray(unsigned long int* arr);
	
	/**
	 * Destructor.
	 */
	virtual ~CCV(){};
	
	//The number of pixels of the associated image
	int m_numPix;

	//The CCV for the r channel
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > m_CCV_r; 

	//The CCV for the g channel
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > m_CCV_g; 

	//The CCV for the b channel
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > m_CCV_b; 

private:
	/**
	* \brief 	Calculates the coherence of each pixel. The
	*		coherence is the size of the pixel's connected
	*		component. So we just have to count the number
	*		of occurrences of the pixel's label.
	* 
	* \param	inputColors	The image belonging to the labeled
	*				connected components
	* \param	inputLabels	The labeled connected components 
	* \return	A std::map containing the size and color value for each
	*		connected component
	*/
	std::map<unsigned short, std::pair<unsigned char, unsigned long> >calcCoherence(cv::Mat inputColors, cv::Mat inputLabels);

	/**
	 * \brief Calculates the CCV for the given image.
	 * 
	 * \param	img			The image to calculate the CCV for
	 * \param	numColors		The number of colors to reduce the image to
	 * \param	coherenceThreshold	A threshold which indicates the minimum size
	 *					of a color region to consider it's pixels as
	 *					coherent
	 *
	 * \return	A color coherence vector given by a std::map that holds the alpha and beta
	 *		values for each color.
	 */
	std::map< unsigned char, std::pair<unsigned long, unsigned long> > calculateCCV(cv::Mat img);

	//The number of colors
	int m_numColors;

	//The coherence threshold
	int m_coherenceThreshold;
};

}

#endif /* CCV_HPP_ */

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

};

}

#endif /* IMAGEPROCESSOR_HPP_ */

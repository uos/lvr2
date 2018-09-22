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
 * Texture.hpp
 *
 *  @date 19.07.2017
 *  @author Jan Philipp Vogtherr (jvogtherr@uos.de)
 *  @author Kristin Schmidt (krschmidt@uos.de)
 */

#ifndef LVR2_TEXTURE_TEXTURE_HPP_
#define LVR2_TEXTURE_TEXTURE_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include <lvr2/geometry/BoundingRectangle.hpp>
#include <lvr2/io/PPMIO.hpp>
#include <lvr/texture/Texture.hpp>

namespace lvr2 
{

class GlTexture;

/**
 * @class Texture
 * @brief This class represents a texture.
 */
class Texture {
public:

    /**
     * @brief Constructor
     */
    Texture( );

    /**
     * @brief Constructor
     */
    Texture(
        int index,
        unsigned short int width,
        unsigned short int height,
        unsigned char numChannels,
        unsigned char numBytesPerChan,
        float texelSize
    );

    /**
     * @brief Constructor
     */
    Texture(
        int index,
        lvr::Texture* oldTexture
    );

    /**
     * @brief Constructor
     */
    Texture(
        int index,
        GlTexture* oldTexture
    );

    /**
     * @brief Constructor
     */
    Texture(Texture&& other);

    /**
     * @brief Constructor
     */
    Texture(const Texture& other);

    Texture & operator=(const Texture &other);

    /**
     * @brief Destructor
     */
    virtual ~Texture();

    /**
     * @brief Writes the texture to an image file
     *
     * The file name will be texture_<INDEX>.ppm and the used file format is Portable Pixel Map (ppm).
     */
    void save();

    /// Texture index
    int m_index;

    /// The dimensions of the texture
    unsigned short int m_width, m_height;

    /// The texture data
    unsigned char* m_data;

    /// The number of color channels
    unsigned char m_numChannels;

    /// The number of bytes per channel
    unsigned char m_numBytesPerChan;

    /// The size of a texture pixel
    float m_texelSize;

};

} // namespace lvr2

#endif /* LVR2_TEXTURE_TEXTURE_HPP_ */

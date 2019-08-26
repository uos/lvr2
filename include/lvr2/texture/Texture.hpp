/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include "lvr2/geometry/BoundingRectangle.hpp"
#include "lvr2/io/PPMIO.hpp"

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
        float texelSize,
        unsigned char* data = nullptr
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

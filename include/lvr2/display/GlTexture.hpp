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

/**
 * Texture.hpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 */

#ifndef LVR2_GLTEXTURE_HPP_
#define LVR2_GLTEXTURE_HPP_

#include <string>
using std::string;

#ifdef _MSC_VER
#include <Windows.h>
#endif

#ifndef __APPLE__
#include <GL/gl.h>
#else
#include <OpenGL/gl.h>
#endif

#include "lvr2/texture/Texture.hpp"

namespace lvr2
{

class GlTexture
{
public:

    /**
     * @brief   Initializes a texture with given date. Class
     *          will release the data.
     *
     * @param   pixels  The image data of the texture. Pixel
     *                  aligned, three bytes per pixel
     * @param   width   The image width
     * @param   height  The image height
     */
    GlTexture(unsigned char* pixels, int width, int height);

    /**
     * @brief   Copy ctor.
     */
    GlTexture(const GlTexture &other);

    /**
     * @brief   Copy ctor.
     */
    GlTexture(const Texture &other);

    /**
     * @brief Empty ctor.
     */
    GlTexture() : m_pixels(0), m_texIndex(0) {};

    /**
     * @brief   Dtor.
     */
    virtual ~GlTexture();

    /**
     * @brief   Bind the texture to the current OpenGL context
     */
    void bind() const { glBindTexture(GL_TEXTURE_2D, m_texIndex);}

    /**
     * @brief   Does all the OpenGL stuff to make it avialable for
     *          rendering.
     */
    void upload();

    /// The width of the texture in pixels
    int                 m_width;

    /// The height of the texture in pixels
    int                 m_height;

    /// The aligned pixel data. Three bytes per pixel (r,g,b)
    unsigned char*      m_pixels;

    /// The texture index of the texture
    GLuint              m_texIndex;
};

} // namespace lvr2

#endif /* TEXTURE_HPP_ */

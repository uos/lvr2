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
 * GlTexture.cpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 */

#include "lvr2/display/GlTexture.hpp"
#include <iostream>

using namespace std;

namespace lvr2
{

GlTexture::GlTexture(unsigned char* pixels, int width, int height)
    : m_width(width), m_height(height),  m_pixels(pixels)
{
    m_texIndex = 0;
    // Upload texture
    upload();
}

GlTexture::GlTexture(const GlTexture &other)
{
    // Copy data
    m_width = other.m_width;
    m_height = other.m_height;
    m_pixels = new unsigned char[3 * m_width * m_height];
    m_texIndex = other.m_texIndex;
    for(int i = 0; i < 3 * m_width * m_height; i++)
    {
        m_pixels[i] = other.m_pixels[i];
    }

    // Upload texture
    upload();
}

GlTexture::GlTexture(const Texture &other)
{
    m_texIndex = 0;
    m_width = other.m_width;
    m_height = other.m_height;
    m_pixels = new unsigned char[3 * m_width * m_height];
    std::fill(m_pixels, m_pixels + 3 * m_width * m_height, 0);

    for (size_t i = 0; i < m_width * m_height; i++)
    {
            size_t current_idx = i * other.m_numChannels * other.m_numBytesPerChan;
            if (other.m_numChannels > 0)
            {
                m_pixels[i*3 + 0] = other.m_data[current_idx + other.m_numBytesPerChan * 0];
            }
            if (other.m_numChannels > 1)
            {
                m_pixels[i*3 + 1] = other.m_data[current_idx + other.m_numBytesPerChan * 1];
            }
            if (other.m_numChannels > 2)
            {
                m_pixels[i*3 + 2] = other.m_data[current_idx + other.m_numBytesPerChan * 2];
            }
    }

    // Upload texture
    upload();
}

GlTexture::~GlTexture()
{
    if(m_pixels)
    {
        delete[] m_pixels;
    }

    if (m_texIndex)
    {
        glDeleteTextures(1, &m_texIndex);
    }
}

void GlTexture::upload()
{
    glEnable(GL_TEXTURE_2D);
    // Create new texure list
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);                              // Set correct alignment
    glGenTextures(1, &m_texIndex);

    /*
    for(int i = 0; i < 3 * m_width * m_height; i++)
      {
        if(i % 3 == 0)
	  {
	    if((int)m_pixels[i] != 0 && (int)m_pixels[i] != 255)
	      cout << endl;
	  }
	if((int)m_pixels[i] != 0 && (int)m_pixels[i] != 255)
	  cout << "GlTexture::upload -- " << (int)m_pixels[i] << " ";
      }
    */
    
    // Bind texture, setup parameters and upload it
    // to video memory
    glBindTexture(GL_TEXTURE_2D, m_texIndex);                           // Bind texture

    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);      // Set repeating and filtering
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);        // Do not apply lighting

    // Upload texture
    glTexImage2D(
		 GL_TEXTURE_2D,
		 0,
		 GL_RGB,
		 m_width,
		 m_height,
		 0,
		 GL_RGB,
		 GL_UNSIGNED_BYTE,
		 m_pixels
		 );
}

} // namespace lvr2

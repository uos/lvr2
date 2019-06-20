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
 * Texture.cpp
 *
 *  @date 19.07.2017
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Kristin Schmidt (krschmidt@uos.de)
 *  @author Jan Philipp Vogtherr (jvogtherr@uos.de)
 */

#include "lvr2/texture/Texture.hpp"
#include "lvr2/display/GlTexture.hpp"

namespace lvr2 {

Texture::Texture()
{
    this->m_data                 = 0;
    this->m_width                = 0;
    this->m_height               = 0;
    this->m_numChannels          = 0;
    this->m_numBytesPerChan      = 0;
    this->m_texelSize            = 1.0;
}

// Texture::Texture(unsigned short int width, unsigned short int height, unsigned char numChannels,
//          unsigned char numBytesPerChan, unsigned short int textureClass, unsigned short int numFeatures,
//          unsigned char numFeatureComponents, float* features, float* keyPoints, float* stats, bool isPattern,
//          unsigned char numCCVColors, unsigned long* CCV)
// {
//     this->m_width                = width;
//     this->m_height               = height;
//     this->m_numChannels          = numChannels;
//     this->m_numBytesPerChan      = numBytesPerChan;
//     this->m_data                 = new unsigned char[width * height * numChannels * numBytesPerChan];
//     this->m_textureClass         = textureClass;
//     this->m_numFeatures          = numFeatures;
//     this->m_numFeatureComponents = numFeatureComponents;
//     this->m_featureDescriptors   = features;
//     this->m_stats                = stats;
//     this->m_isPattern            = isPattern;
//     this->m_numCCVColors         = numCCVColors;
//     this->m_CCV                  = CCV;
//     this->m_keyPoints            = keyPoints;
//     this->m_distance             = 0;
// }

Texture::Texture(
    int index,
    unsigned short int width,
    unsigned short int height,
    unsigned char numChannels,
    unsigned char numBytesPerChan,
    float texelSize,
    unsigned char* data
) :
    m_index(index),
    m_width(width),
    m_height(height),
    m_numChannels(numChannels),
    m_numBytesPerChan(numBytesPerChan),
    m_texelSize(texelSize),
    m_data(new unsigned char[width * height * numChannels * numBytesPerChan])
{
    if(data)
    {
        memcpy(m_data, data, width * height * numChannels * numBytesPerChan);
    }
}

Texture::Texture(Texture &&other) {
    this->m_index = other.m_index;
    this->m_width = other.m_width;
    this->m_height = other.m_height;
    this->m_data = other.m_data;
    this->m_numChannels = other.m_numChannels;
    this->m_numBytesPerChan = other.m_numBytesPerChan;
    this->m_texelSize = other.m_texelSize;

    other.m_data = nullptr;
    other.m_width = 0;
    other.m_height = 0;
    other.m_numChannels = 0;
    other.m_numBytesPerChan = 0;
}

Texture::Texture(const Texture& other)
{
    this->m_index = other.m_index;
    this->m_width = other.m_width;
    this->m_height = other.m_height;
    //this->m_data = other.m_data;
    this->m_numChannels = other.m_numChannels;
    this->m_numBytesPerChan = other.m_numBytesPerChan;
    this->m_texelSize = other.m_texelSize;

    size_t data_size = m_width * m_height * m_numChannels * m_numBytesPerChan;
    m_data = new unsigned char[data_size];
    std::copy(other.m_data, other.m_data + data_size, m_data);
    
}

Texture & Texture::operator=(const Texture &other)
{
    if (this != &other)
    {

        if (m_data)
        {
            delete[] m_data;
        }
        this->m_index = other.m_index;
        this->m_width = other.m_width;
        this->m_height = other.m_height;
        //this->m_data = other.m_data;
        this->m_numChannels = other.m_numChannels;
        this->m_numBytesPerChan = other.m_numBytesPerChan;
        this->m_texelSize = other.m_texelSize;

        size_t data_size = m_width * m_height * m_numChannels * m_numBytesPerChan;
        m_data = new unsigned char[data_size];
        std::copy(other.m_data, other.m_data + data_size, m_data);
    }

    return *this;
}


Texture::Texture(
    int index,
    GlTexture* oldTexture
) :
    m_index(index),
    m_width(oldTexture->m_width),
    m_height(oldTexture->m_height),
    m_numChannels(3),
    m_numBytesPerChan(sizeof(unsigned char)),
    m_texelSize(1),
    m_data(new unsigned char[oldTexture->m_width
                    * oldTexture->m_height
                    * 3
                    * sizeof(unsigned char)])
{
    size_t copy_len = m_width * m_height * m_numChannels * m_numBytesPerChan;
    std::copy(oldTexture->m_pixels, oldTexture->m_pixels + copy_len, m_data);
}

void Texture::save()
{
    //write image file
    char fn[255];
    sprintf(fn, "texture_%d.ppm", m_index);
    PPMIO* pio = new PPMIO;
    pio->setDataArray(this->m_data, m_width, m_height);
    pio->write(string(fn));
    delete pio;
}

Texture::~Texture() {
    if (m_data)
    {
        delete[] m_data;
    }
    // delete[] m_featureDescriptors;
    // delete[] m_stats;
    // delete[] m_CCV;
    // delete[] m_keyPoints;
}

}

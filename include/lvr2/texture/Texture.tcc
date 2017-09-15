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
 * Texture.cpp
 *
 *  @date 19.07.2017
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Kristin Schmidt (krschmidt@uos.de)
 *  @author Jan Philipp Vogtherr (jvogtherr@uos.de)
 */


namespace lvr2 {

template<typename BaseVecT>
Texture<BaseVecT>::Texture()
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

template<typename BaseVecT>
Texture<BaseVecT>::Texture(
    int index,
    unsigned short int width,
    unsigned short int height,
    unsigned char numChannels,
    unsigned char numBytesPerChan,
    float texelSize
) :
    m_index(index),
    m_width(width),
    m_height(height),
    m_numChannels(numChannels),
    m_numBytesPerChan(numBytesPerChan),
    m_texelSize(texelSize),
    m_data(new unsigned char[width * height * numChannels * numBytesPerChan])
{
}

template<typename BaseVecT>
void Texture<BaseVecT>::save()
{
    //write image file
    char fn[255];
    sprintf(fn, "texture_%d.ppm", m_index);
    lvr::PPMIO* pio = new lvr::PPMIO;
    pio->setDataArray(this->m_data, m_width, m_height);
    pio->write(string(fn));
    delete pio;
}

template<typename BaseVecT>
Texture<BaseVecT>::~Texture() {
    // delete[] m_data;
    // delete[] m_featureDescriptors;
    // delete[] m_stats;
    // delete[] m_CCV;
    // delete[] m_keyPoints;
}

}

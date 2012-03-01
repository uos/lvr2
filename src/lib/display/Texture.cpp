/**
 * Texture.cpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 */

#include "Texture.hpp"

Texture::Texture(unsigned char* pixels, int width, int height)
    : m_width(width), m_height(height),  m_pixels(pixels)
{
    m_texIndex = 0;
    // Upload texture
    upload();
}

Texture::Texture(const Texture &other)
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

#include <iostream>
using namespace std;

void Texture::upload()
{
    glEnable(GL_TEXTURE_2D);
    // Create new texure list
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);                              // Set correct alignment
    glGenTextures(1, &m_texIndex);

    cout << "TEX INDEX: " << m_texIndex << endl;

    for(int i = 0; i < 3 * m_width * m_height; i++)
    {
        cout << (int)m_pixels[i] << " ";
        if(i % 3 == 0) cout << endl;
    }

    // Bind texture, setup parameters and upload it
    // to video memory
    glBindTexture(GL_TEXTURE_2D, m_texIndex);                           // Bind texture

    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);      // Set repeating and filetering
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);           // Do not apply lighting

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
            m_pixels);

}

Texture::~Texture()
{
    if(m_pixels)
    {
        delete[] m_pixels;
    }
}


/**
 * Texture.hpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 */

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include <string>
using std::string;

#include <GL/gl.h>

class Texture
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
    Texture(unsigned char* pixels, int width, int height);

    /**
     * @brief   Copy ctor.
     */
    Texture(const Texture &other);

    /**
     * @brief   Dtor.
     */
    virtual ~Texture();

    /**
     * @brief   Bind the texture to the current OpenGL context
     */
    void bind() const { glBindTexture(GL_TEXTURE_2D, m_texIndex);}

private:

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

#endif /* TEXTURE_HPP_ */

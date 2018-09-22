/**
 * Texture.hpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 */

#ifndef GLTEXTURE_HPP_
#define GLTEXTURE_HPP_

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

#include <lvr2/texture/Texture.hpp>

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
    GlTexture() {};

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

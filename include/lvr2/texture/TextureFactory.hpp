/**
 * TextureFactory.h
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 *  @date 15.02.2012
 *  @author Denis Meyer
 */

#ifndef TEXTUREFACTORY_H_
#define TEXTUREFACTORY_H_

#include <string>

namespace lvr2
{

class Texture;

class TextureFactory
{

public:

    /**
     * @brief   Returns a new texture if the file contains readable
     *          image data or a null point if the file couldn't be parsed
     */
    static Texture readTexture(std::string filename);

    /**
     * @brief   TODO
     */
    static void saveTexture(const Texture& texture, std::string filename);
};

} // namespace lvr2

#endif /* TEXTUREFACTORY_H_ */

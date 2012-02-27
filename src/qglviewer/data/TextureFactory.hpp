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

#include "rendering/Texture.hpp"

class TextureFactory
{
protected:

    /**
     * @brief Constructor
     */
    TextureFactory();

    /**
     * @brief Destructor
     */
    virtual ~TextureFactory();

public:

    /**
     * @brief   Returns a new texture if the file contains readable
     *          image data or a null point if the file couldn't be parsed
     */
    Texture* getTexture(string filename) const;

    /**
     * @brief   Returns the singleton instance
     */
    static TextureFactory& instance();
};

#endif /* TEXTUREFACTORY_H_ */

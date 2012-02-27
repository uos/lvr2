/**
 * TextureFactory.cpp
 *
 *  @date 11.12.2011
 *  @author Thomas Wiemann
 *  @date 15.02.2012
 *  @author Denis Meyer
 */

#include "TextureFactory.hpp"
#include "ReadPPM.hpp"

#include <iostream>
using std::cout;
using std::endl;

TextureFactory::TextureFactory()
{
    // TODO Auto-generated constructor stub

}

TextureFactory::~TextureFactory()
{
    // TODO Auto-generated destructor stub
}

TextureFactory& TextureFactory::instance()
{
    // Just create on instance
    static TextureFactory instance;
    return instance;
}

Texture* TextureFactory::getTexture(string filename) const
{
    // A texture object
    Texture* tex = 0;

    // Texture data
    int width = 0;
    int height = 0;
    unsigned char* data = 0;

    // Get file extension
    if(filename.substr(filename.find_last_of(".") + 1) == "ppm")
    {
        ReadPPM reader(filename);
        data    = reader.getPixels();
        width   = reader.getWidth();
        height  = reader.getHeight();
    }

    // Check data and create new texture if possible
    if(data != 0 && width != 0 && height != 0)
    {
        tex = new Texture(data, width, height);
    }
    else
    {
        cout << "TextureFactory: Unable to read file " << filename << "." << endl;
    }

    return tex;
}

#include "RayCastingTexturizer.hpp"

namespace lvr2
{

template <typename BaseVecT>
RayCastingTexturizer<BaseVecT>::RayCastingTexturizer(
    float texelMinSize,
    int texMinClusterSize,
    int texMaxClusterSize
): Texturizer<BaseVecT>(texelMinSize, texMinClusterSize, texMaxClusterSize)
{

}

template <typename BaseVecT>
TextureHandle RayCastingTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>&,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
)
{

    // Calculate the texture size
    unsigned short int sizeX = ceil(abs(boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil(abs(boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);
    
    Texture tex(
        index,
        sizeX,
        sizeY,
        3,
        1,
        this->m_texelSize
    );
    tex.m_layerName = "RGB";

    size_t num_pixel = sizeX * sizeY;

    // Init red
    for (int i = 0; i < num_pixel; i++)
    {
        tex.m_data[i * 3 + 0] = 255;
        tex.m_data[i * 3 + 1] = 0;
        tex.m_data[i * 3 + 2] = 0;
    }

    return this->m_textures.push(tex);
}




} // namespace lvr2


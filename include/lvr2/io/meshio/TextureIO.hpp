#pragma once

#include <lvr2/io/meshio/FeatureBase.hpp>
#include <lvr2/texture/Texture.hpp>

namespace lvr2
{

template <typename FeatureBase>
class TextureIO
{
public:
    void saveTexture(
        const std::string& mesh_name,
        const size_t material_index,
        const std::string& texture_name,
        const MeshBufferPtr& mesh,
        const Texture& texture
    ) const;
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<TextureIO, FeatureBase>
{
    using type = typename FeatureBase::template add_features<TextureIO>::type;
};

} // namespace lvr2

#include "TextureIO.tcc"
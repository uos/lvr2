#pragma once

#include <lvr2/io/meshio/FeatureBase.hpp>
#include <lvr2/io/meshio/TextureIO.hpp>
#include <lvr2/texture/Material.hpp>

namespace lvr2
{

using MaterialOptional = boost::optional<Material>;
using TextureOptional = boost::optional<Texture>;

template <typename FeatureBase>
class MaterialIO
{
public:
    // TODO: Support multiple textures per material
    void saveMaterial(
        const std::string& mesh_name,
        const size_t& material_index,
        const MeshBufferPtr& mesh
    ) const;

    std::pair<MaterialOptional, TextureOptional> loadMaterial(
        const std::string& mesh_name,
        const size_t& material_index
    ) const;
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    TextureIO<FeatureBase>* m_textureIO 
        = static_cast<TextureIO<FeatureBase>*>(m_featureBase);   

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<MaterialIO, FeatureBase>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<TextureIO, FeatureBase>::type;

    // Add the feature
    using type = typename dep1::template add_features<MaterialIO>::type;
};


} // namespace lvr2

#include "MaterialIO.tcc"
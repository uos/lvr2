#include "TextureIO.hpp"
#include <numeric>
#include <functional>
#include <lvr2/io/meshio/yaml/Texture.hpp>

namespace lvr2
{

template <typename FeatureBase>
void TextureIO<FeatureBase>::saveTexture(   
    const std::string& mesh_name,
    const size_t material_index,
    const std::string& tex_name,
    const Texture& tex
) const
{
    Description desc = m_featureBase->m_schema->texture(mesh_name, material_index, tex_name);

    size_t byte_count = tex.m_width * tex.m_height * tex.m_numChannels * tex.m_numBytesPerChan;
    ucharArr copy(new uint8_t[byte_count]);
    std::copy(
        tex.m_data,
        tex.m_data + byte_count,
        copy.get());

    m_featureBase->m_kernel->saveUCharArray(
        *desc.dataRoot,
        *desc.data,
        {tex.m_height, 
        tex.m_width, 
        tex.m_numChannels, 
        tex.m_numBytesPerChan},
        copy);

    // Save metadata
    YAML::Node meta;
    meta = tex;
    m_featureBase->m_kernel->saveMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );
}

template <typename FeatureBase>
Texture TextureIO<FeatureBase>::loadTexture(
    const std::string& mesh_name,
    const size_t material_index,
    const std::string& texture_name
) const
{
    Texture ret;
    Description desc = m_featureBase->m_schema->texture(
        mesh_name,
        material_index,
        texture_name
    );

    // Load meta
    YAML::Node meta;
    m_featureBase->m_kernel->loadMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );
    // Decode meta 
    // This allocates the Texture memory because it calles the Texture Constructor
    ret = meta.as<lvr2::Texture>();

    std::vector<size_t> dims = {
        ret.m_height, 
        ret.m_width, 
        ret.m_numChannels, 
        ret.m_numBytesPerChan
        };

    ucharArr data = m_featureBase->m_kernel->loadUCharArray(
        *desc.dataRoot,
        *desc.data,
        dims
    );
    // Calculate 1d Array length
    size_t byte_count = std::accumulate(
        dims.begin(),
        dims.end(),
        1,
        std::multiplies<size_t>()
    );
    // Copy texture data
    std::copy(
        data.get(), // begin iterator
        &(data.get()[byte_count]), // past-the-end iterator
        ret.m_data
    );    

    return std::move(ret);
}
} // namespace lvr2
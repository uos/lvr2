#include "TextureIO.hpp"

namespace lvr2
{

template <typename FeatureBase>
void TextureIO<FeatureBase>::saveTexture(   
    const std::string& mesh_name,
    const size_t material_index,
    const std::string& tex_name,
    const MeshBufferPtr& mesh,
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

} // namespace lvr2
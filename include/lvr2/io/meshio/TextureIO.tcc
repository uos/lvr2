#include "TextureIO.hpp"
#include <numeric>
#include <functional>
#include <lvr2/io/meshio/yaml/Texture.hpp>
#include <opencv2/core/types_c.h>

namespace lvr2
{

namespace meshio
{

template <typename BaseIO>
void TextureIO<BaseIO>::saveTexture(   
    const std::string& mesh_name,
    const size_t material_index,
    const std::string& tex_name,
    const Texture& tex
) const
{
    Description desc = m_baseIO->m_description->texture(mesh_name, material_index, tex_name);

    size_t byte_count = tex.m_width * tex.m_height * tex.m_numChannels * tex.m_numBytesPerChan;
    ucharArr copy(new uint8_t[byte_count]);
    std::copy(
        tex.m_data,
        tex.m_data + byte_count,
        copy.get());
    
    if (byte_count == 0)
    {
        std::cout << timestamp << "[TextureIO] Texture with index: '" << tex.m_index 
            << "' has byte_count '0'. w: " << tex.m_width 
            << " h: " << tex.m_height
            << " n channels: " << tex.m_numChannels
            << " channel width: " << tex.m_numBytesPerChan << std::endl;
        std::cout << timestamp << "[TextureIO] Skipping texture" << std::endl;
        return;
    }
    m_baseIO->m_kernel->saveUCharArray(
        *desc.dataRoot,
        *desc.data,
        {tex.m_height, 
        tex.m_width, 
        tex.m_numChannels, 
        tex.m_numBytesPerChan},
        copy);

    // TODO: save textures as via kernel->saveImage()
    // cv::Mat matrix(
    //     {tex.m_height,
    //     tex.m_width,
    //     tex.m_numChannels},
    //     CV_8UC1,
    //     copy.get()
    // );

    // m_baseIO->m_kernel->saveImage(
    //     *desc.dataRoot,
    //     *desc.data,
    //     matrix
    // );

    // Save metadata
    YAML::Node meta;
    meta = tex;
    m_baseIO->m_kernel->saveMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );
}

template <typename BaseIO>
TextureOptional TextureIO<BaseIO>::loadTexture(
    const std::string& mesh_name,
    const size_t material_index,
    const std::string& texture_name
) const
{
    Texture ret;
    Description desc = m_baseIO->m_description->texture(
        mesh_name,
        material_index,
        texture_name
    );

    // Check if exists
    if (!m_baseIO->m_kernel->exists(*desc.dataRoot, *desc.data))
    {
        return TextureOptional();
    }

    // Load meta
    YAML::Node meta;
    m_baseIO->m_kernel->loadMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );
    // Decode meta 
    // This allocates the Texture memory because it calles the Texture Constructor
    ret = meta.as<lvr2::Texture>();
    ret.m_layerName = texture_name;

    std::vector<size_t> dims = {
        ret.m_height, 
        ret.m_width, 
        ret.m_numChannels, 
        ret.m_numBytesPerChan
        };

    ucharArr data = m_baseIO->m_kernel->loadUCharArray(
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

} // namespace meshio
} // namespace lvr2
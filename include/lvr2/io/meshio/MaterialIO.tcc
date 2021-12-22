#include "MaterialIO.hpp"
#include <lvr2/io/meshio/yaml/Material.hpp>

namespace lvr2
{

template <typename FeatureBase>
void MaterialIO<FeatureBase>::saveMaterial(
    const std::string& mesh_name,
    const size_t& material_index,
    const MeshBufferPtr& mesh
) const
{
    const auto& materials = mesh->getMaterials();
    const auto& textures = mesh->getTextures();
    
    Description desc = m_featureBase->m_schema->material(mesh_name, material_index);
    const Material& mat = materials[material_index];

    // Write metadata
    YAML::Node meta;
    meta = mat;
    
    m_featureBase->m_kernel->saveMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );

    // Write Textures TODO: Add support for multiple layers
    if (mat.m_texture)
    {
        const Texture tex = textures[(*mat.m_texture).idx()];
        m_textureIO->saveTexture(
            mesh_name,
            material_index,
            "RGB",
            tex
        );
    }

}

template <typename FeatureBase>
std::pair<MaterialOptional, TextureOptional> MaterialIO<FeatureBase>::loadMaterial(
        const std::string& mesh_name,
        const size_t& material_index
    ) const
{
    Description desc = m_featureBase->m_schema->material(
        mesh_name,
        material_index
    );
    // Check if material exists
    if (!m_featureBase->m_kernel->exists(*desc.dataRoot))
    {
        return std::make_pair(MaterialOptional(), TextureOptional());
    }

    YAML::Node node;
    m_featureBase->m_kernel->loadMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        node
    );
    Material ret_mat = node.as<lvr2::Material>();

    TextureOptional ret_tex_opt;
    // Get all texture names
    std::vector<std::string> textures;  
    m_featureBase->m_kernel->subGroupNames(
            *desc.dataRoot + "/textures",
            textures
        );
    if (textures.size() > 0)
    {
        ret_tex_opt = m_featureBase->m_textureIO->loadTexture(
            mesh_name,
            material_index,
            textures[0]
        );
    }

    return std::make_pair(ret_mat, ret_tex_opt);
}

} // namespace lvr2
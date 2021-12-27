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
            tex.m_layerName,
            tex
        );
    }

}

template <typename FeatureBase>
std::pair<MaterialOptional, TextureOptional> MaterialIO<FeatureBase>::loadMaterial(
        const std::string& mesh_name,
        const size_t& material_index,
        StringOptional texture_layer
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
    if (m_featureBase->m_kernel->exists(*desc.dataRoot + "/textures"))
    {
        // Get all texture names
        std::vector<std::string> textures;  
        m_featureBase->m_kernel->subGroupNames(
                *desc.dataRoot + "/textures",
                textures
            );
        if (!textures.empty())
        {
            std::string layer;

            if (texture_layer)
            {
                layer = *texture_layer;
            }
            else
            {
                layer = textures[0];
            }

            ret_tex_opt = m_featureBase->m_textureIO->loadTexture(
                mesh_name,
                material_index,
                layer
            );
        }
    }

    return std::make_pair(ret_mat, ret_tex_opt);
}

template <typename FeatureBase>
size_t MaterialIO<FeatureBase>::loadMaterials(
    const std::string& mesh_name,
    MeshBufferPtr mesh,
    StringOptional texture_layer) const
{
    size_t count = 0;
    while(true)
    {
        Description desc = m_featureBase->m_schema->material(mesh_name, count);
        if (!m_featureBase->m_kernel->exists(
            *desc.dataRoot,
            *desc.data
        )) break;
        count++;
    }
    // Clear old materials before loading new ones
    mesh->getMaterials().clear();
    ProgressBar bar(count, timestamp.getElapsedTime() + "[MeshIO] Loading materials & textures");
    size_t material_idx = 0;
    while(true)
    {
        auto res = loadMaterial(
            mesh_name,
            material_idx,
            texture_layer
        );
        // Check if a Material was loaded
        if (!res.first)
        {
            break;
        }

        
        // Check if a Texture was loaded
        if (res.second)
        {
            mesh->getTextures()[res.second->m_index] = *res.second;
        }
        else
        {
            res.first->m_texture = boost::optional<TextureHandle>();
        }

        mesh->getMaterials().push_back(*res.first);

        ++bar;
        ++material_idx;
    }
    std::cout << std::endl;

    return mesh->getMaterials().size();
}

template <typename FeatureBase>
void MaterialIO<FeatureBase>::saveMaterials(
        const std::string& mesh_name,
        const MeshBufferPtr& mesh
    ) const
{
    const auto& materials = mesh->getMaterials();
    ProgressBar material_progress( materials.size(), timestamp.getElapsedTime() + "[MeshIO] Saving materials & textures");
    for (size_t idx = 0; idx < materials.size(); idx++)
    {
        saveMaterial(
            mesh_name,
            idx,
            mesh
        );
        ++material_progress;
    }
    std::cout << std::endl;
}

} // namespace lvr2
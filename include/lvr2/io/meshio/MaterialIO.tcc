#include "MaterialIO.hpp"

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
            mesh,
            tex
        );
    }

}

} // namespace lvr2
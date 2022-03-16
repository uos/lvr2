#pragma once

#include <lvr2/io/baseio/BaseIO.hpp>
#include <lvr2/io/meshio/TextureIO.hpp>
#include <lvr2/texture/Material.hpp>

using lvr2::baseio::FeatureConstruct;

namespace lvr2
{
namespace meshio
{
    using MaterialOptional = boost::optional<Material>;
    using TextureVector = std::vector<Texture>;
    using TextureVectorOpt = boost::optional<TextureVector>;

    template <typename BaseIO>
    class MaterialIO
    {
    public:
        void saveMaterial(
            const std::string& mesh_name,
            const size_t& material_index,
            const MeshBufferPtr& mesh
        ) const;

        void saveMaterials(
            const std::string& mesh_name,
            const MeshBufferPtr& mesh
        ) const;

        std::pair<MaterialOptional, TextureVectorOpt> loadMaterial(
            const std::string& mesh_name,
            const size_t& material_index
        ) const;

        /**
     * @brief Loads all Materials associated with \p mesh_name
     *
     * @param mesh_name The name of the Mesh in the h5 file
     * @param[out] mesh The Materials and textures will be added to this mesh. \p mesh needs to have enough memory allocated to hold all textures
     * @return The number of Materials loaded
     */
        size_t loadMaterials(const std::string& mesh_name, MeshBufferPtr mesh) const;
    protected:
        BaseIO* m_baseIO = static_cast<BaseIO*>(this);
        TextureIO<BaseIO>* m_textureIO = static_cast<TextureIO<BaseIO>*>(m_baseIO);

    };

} // namespace meshio

template <typename FB>
struct FeatureConstruct<lvr2::meshio::MaterialIO, FB>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<lvr2::meshio::TextureIO, FB>::type;

    // Add the feature
    using type = typename dep1::template add_features<lvr2::meshio::MaterialIO>::type;
};


} // namespace lvr2

#include "MaterialIO.tcc"
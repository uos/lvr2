#pragma once

#include <lvr2/io/baseio/BaseIO.hpp>
#include <lvr2/texture/Texture.hpp>

namespace lvr2
{
namespace meshio
{

    using TextureOptional = boost::optional<Texture>;

    template <typename BaseIO>
    class TextureIO
    {
    public:
        void saveTexture(
            const std::string &mesh_name,
            const size_t material_index,
            const std::string &texture_name,
            const Texture &texture) const;

        TextureOptional loadTexture(
            const std::string &mesh_name,
            const size_t material_index,
            const std::string &texture_name) const;

    protected:
        BaseIO *m_baseIO = static_cast<BaseIO *>(this);
    };

} // namespace meshio

template <typename FB>
struct lvr2::baseio::FeatureConstruct<lvr2::meshio::TextureIO, FB>
{
    using type = typename FB::template add_features<lvr2::meshio::TextureIO>::type;
};

} // namespace lvr2

#include "TextureIO.tcc"
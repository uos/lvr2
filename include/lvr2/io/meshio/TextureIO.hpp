#pragma once

#include <lvr2/io/scanio/FeatureBase.hpp>
#include <lvr2/texture/Texture.hpp>

namespace lvr2
{
    namespace meshio {

        using TextureOptional = boost::optional<Texture>;

        template <typename FeatureBase>
        class TextureIO
        {
        public:
            void saveTexture(
                const std::string& mesh_name,
                const size_t material_index,
                const std::string& texture_name,
                const Texture& texture
            ) const;

            TextureOptional loadTexture(
                const std::string& mesh_name,
                const size_t material_index,
                const std::string& texture_name
            ) const;
        protected:
            FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

        };

template <typename FB>
struct FeatureConstruct<TextureIO, FB>
{
    using type = typename FB::template add_features<TextureIO>::type;
};

} // namespace meshio
} // namespace lvr2

#include "TextureIO.tcc"
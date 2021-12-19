#pragma once

#include <lvr2/io/meshio/FeatureBase.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MaterialIO.hpp>

namespace lvr2
{

template <typename FeatureBase>
class MeshIO
{
public:
    void saveMesh(
        std::string mesh_name, 
        MeshBufferPtr mesh
        ) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    MaterialIO<FeatureBase>* m_materialIO 
        = static_cast<MaterialIO<FeatureBase>*>(m_featureBase);

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<MeshIO, FeatureBase>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<MaterialIO, FeatureBase>::type;

    // Add the feature
    using type = typename dep1::template add_features<MeshIO>::type;
};

} // namespace lvr2

#include "MeshIO.tcc"
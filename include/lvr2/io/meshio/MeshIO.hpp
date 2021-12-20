#pragma once

#include <lvr2/io/meshio/FeatureBase.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MaterialIO.hpp>
#include <lvr2/io/meshio/ClusterIO.hpp>

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

    ClusterIO<FeatureBase>* m_clusterIO 
        = static_cast<ClusterIO<FeatureBase>*>(m_featureBase);

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<MeshIO, FeatureBase>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<MaterialIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<ClusterIO, FeatureBase>::type;

    using deps = typename dep1::template Merge<dep2>;

    // Add the feature
    using type = typename deps::template add_features<MeshIO>::type;
};

} // namespace lvr2

#include "MeshIO.tcc"
#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MeshSchema.hpp>

namespace lvr2
{

template <typename FeatureBase>
class ClusterIO
{
public:
    void saveCluster(
        const std::string& mesh_name,
        const size_t& cluster_idx,
        const MeshBufferPtr& mesh,
        const IndexChannel& index_channel
    );
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
};

} // namespace lvr2

#include "ClusterIO.tcc"
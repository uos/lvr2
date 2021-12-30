#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MeshSchema.hpp>
#include <lvr2/io/DataStruct.hpp>

namespace lvr2
{

struct ClusterIOData
{
    size_t num_faces = 0;
    size_t material_index = 0;
    indexArray face_indices = nullptr;
};

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

    boost::optional<ClusterIOData> loadCluster(
        const std::string& mesh_name,
        const size_t& cluster_idx,
        floatArr tex_coords,
        std::vector<indexArray::element_type>& faces,
        std::vector<indexArray::element_type>& faceToMaterial
    ) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
};

} // namespace lvr2

#include "ClusterIO.tcc"
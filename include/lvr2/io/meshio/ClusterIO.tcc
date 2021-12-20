#include "ClusterIO.hpp"

namespace lvr2
{

template <typename FeatureBase>
void ClusterIO<FeatureBase>::saveCluster(
    const std::string& mesh_name,
    const size_t& cluster_idx,
    const MeshBufferPtr& mesh,
    const IndexChannel& index_channel
)
{
    // Get data needed for saving clusters
    IndexChannelOptional clm_map_opt = mesh->getIndexChannel("cluster_material_indices");
    if (!clm_map_opt)
    {
        std::cout << timestamp << "[MeshIO] No cluster -> material_index channel in buffer!" << std::endl;
    }

    // Texture coordinates for each vertex
    floatArr uv_coords = mesh->getTextureCoordinates();
    if (!uv_coords)
    {
        std::cout << timestamp << "[MeshIO] No uv coordinates in buffer!" << std::endl;
    }


    ////            ////
    //  Prepare Data  //
    ////            ////

    // continuous index buffer for the clusters faces
    std::vector<IndexChannelOptional::value_type::DataType> idx_buffer;
    // continuous index buffer for the text coords
    std::vector<float> uv_buffer;
    

    // Add all face indices to continuous buffer
    // clfi is an index into the clusters face array
    for (size_t clf_i = 0; clf_i < index_channel.numElements(); clf_i++)
    {
        auto face_idx = index_channel[clf_i];
        idx_buffer.push_back(mesh->getFaceIndices()[face_idx * 3 + 0]);
        idx_buffer.push_back(mesh->getFaceIndices()[face_idx * 3 + 1]);
        idx_buffer.push_back(mesh->getFaceIndices()[face_idx * 3 + 2]);
    }
    
    // If uv_coords exist add them to output data
    if (uv_coords)
    {
        // Add uv coordinates for each index to the uv buffer
        for (auto idx: idx_buffer)
        {
            uv_buffer.push_back(uv_coords[idx * 2 + 0]);
            uv_buffer.push_back(uv_coords[idx * 2 + 1]);
        }
    }
    

    ////          ////
    //  Write Data  //
    ////          ////

    // Write index buffer
    Description desc = m_featureBase->m_schema->surfaceIndices(mesh_name, cluster_idx);
    m_featureBase->m_kernel->saveArray(
        *desc.dataRoot,
        *desc.data,
        {idx_buffer.size()},
        Util::convert_vector_to_shared_array(idx_buffer)
        );
    {
        YAML::Node meta;
        meta["data_type"] = "uint32";
        meta["entity"]  = "channel";
        meta["type"]    = "array";
        meta["name"]    = "indices";
        meta["shape"].push_back(idx_buffer.size());
        meta["shape"].push_back(1);
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

    // Write uv buffer
    desc = m_featureBase->m_schema->textureCoordinates(mesh_name, cluster_idx);
    m_featureBase->m_kernel->saveArray(
        *desc.dataRoot,
        *desc.data,
        {idx_buffer.size(), 2}, // 2 float coords for each index
        Util::convert_vector_to_shared_array(uv_buffer)
        );
    {
        YAML::Node meta;
        meta["data_type"] = "float";
        meta["entity"]  = "channel";
        meta["type"]    = "array";
        meta["name"]    = "texture_coordinates";
        meta["shape"].push_back(idx_buffer.size());
        meta["shape"].push_back(2);
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

    // Write surface meta
    desc = m_featureBase->m_schema->surface(mesh_name, cluster_idx);
    YAML::Node meta;
    if (clm_map_opt)
    {
        IndexChannel::DataType idx = (*clm_map_opt)[cluster_idx];
        meta["material"] = idx;
    }
    else
    {
        std::cout << timestamp << "[MeshIO] Cannot add material to surface!" << std::endl;
    }
    m_featureBase->m_kernel->saveMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        meta
    );
    
}

} // namespace lvr2
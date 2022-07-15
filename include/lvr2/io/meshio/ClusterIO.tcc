#include "ClusterIO.hpp"
#include <lvr2/util/Util.hpp>
#include <algorithm>
#include <boost/smart_ptr/make_shared_array.hpp>

namespace lvr2
{
namespace meshio
{
template <typename BaseIO>
void ClusterIO<BaseIO>::saveClusters(
    const std::string& mesh_name,
    const MeshBufferPtr mesh)
{
    // Create one continuous array of clusters
    
    // Contains all face indices 
    std::vector<IndexChannel::DataType> combined_face_indices;
    // 2D vec contains begin and end index in the combined_face_indices buffer
    std::vector<size_t> cluster_ranges;

    size_t cluster_idx = 0;
    while(true)
    {
        // Naming structure see FinalizeAlgorithms.tcc TextureFinalizer::apply
        std::string cluster_name = "cluster" + std::to_string(cluster_idx) + "_face_indices";
        // Contains all face indices of the current cluster
        IndexChannelOptional    clf_idx_opt = mesh->getIndexChannel(cluster_name);

        // If no channel with index surface_idx is found end loop
        if (!clf_idx_opt) break;

        // First face in the cluster
        size_t cluster_start = combined_face_indices.size();

        // Add faces to buffer
        combined_face_indices.insert(
            combined_face_indices.end(),
            clf_idx_opt->dataPtr().get(),
            clf_idx_opt->dataPtr().get() + clf_idx_opt->numElements()
            );

        // Past the end of the sequence
        size_t cluster_end = combined_face_indices.size();
        cluster_ranges.push_back(cluster_start);
        cluster_ranges.push_back(cluster_end);

        cluster_idx++;
    }
    
    // If no clusters exist we can stop here
    if (combined_face_indices.size() <= 0 || cluster_ranges.size() <= 0)
    {
        return;
    }

    // TODO: Meta data
    // Save combined cluster indices
    std::vector<size_t> shape = {combined_face_indices.size(), 1};
    Description desc = m_baseIO->m_description->surfaceCombinedFaceIndices(mesh_name);
    m_baseIO->m_kernel->saveArray(
        *desc.dataRoot,
        *desc.data,
        shape,
        Util::convert_vector_to_shared_array(combined_face_indices)
        );
    {
        YAML::Node meta;
        meta["shape"] = shape;
        m_baseIO->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

    // Save cluster face index ranges
    shape = {cluster_idx, 2};
    desc = m_baseIO->m_description->surfaceFaceIndexRanges(mesh_name);
    m_baseIO->m_kernel->saveArray(
        *desc.dataRoot,
        *desc.data,
        shape,
        Util::convert_vector_to_shared_array(cluster_ranges)
        );
    {
        YAML::Node meta;
        meta["shape"] = shape;
        m_baseIO->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

    // Save cluster material indices
    auto channel_opt = mesh->getIndexChannel("cluster_material_indices");

    if (channel_opt)
    {
        Description desc = m_baseIO->m_description->surfaceMaterialIndices(mesh_name);
        std::vector<size_t> shape = {channel_opt->numElements(), channel_opt->width()};

        m_baseIO->m_kernel->saveArray(
        *desc.dataRoot,
        *desc.data,
        shape,
        channel_opt->dataPtr()
        );

        YAML::Node meta;
        meta["shape"] = shape;
        m_baseIO->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

}


template <typename BaseIO>
void ClusterIO<BaseIO>::loadClusters(
        const std::string& mesh_name,
        MeshBufferPtr mesh
    )
{
    std::vector<size_t> shape;
    // Load combined face indices
    Description desc = m_baseIO->m_description->surfaceCombinedFaceIndices(mesh_name);
    indexArray combined_face_indices = m_baseIO->m_kernel->template loadArray<indexArray::element_type>(
        *desc.dataRoot,
        *desc.data,
        shape
    );

    desc = m_baseIO->m_description->surfaceFaceIndexRanges(mesh_name);
    indexArray face_index_ranges = m_baseIO->m_kernel->template loadArray<indexArray::element_type>(
        *desc.dataRoot,
        *desc.data,
        shape
    );
    // If the Mesh has no clusters return
    if (!(face_index_ranges && combined_face_indices))
    {
        return;
    }

    // Add each cluster to the buffer
    for (size_t i = 0; i < shape[0]; i++)
    {
        size_t cluster_begin    = face_index_ranges[i * 2 + 0];
        size_t cluster_end      = face_index_ranges[i * 2 + 1];
        size_t cluster_length   = cluster_end - cluster_begin;

        indexArray cluster(new indexArray::element_type[cluster_length]);
        // Copy the face indices belonging to this cluster
        std::copy(
            combined_face_indices.get() + cluster_begin,
            combined_face_indices.get() + cluster_end,
            cluster.get()
            );
        // The string id of the cluser
        std::string cluster_name = "cluster" + std::to_string(i) + "_face_indices";

        mesh->addIndexChannel(
            cluster,
            cluster_name,
            cluster_length,
            1
        );
    }

    desc = m_baseIO->m_description->surfaceMaterialIndices(mesh_name);
    if (m_baseIO->m_kernel->exists(*desc.dataRoot, *desc.data))
    {
        std::vector<size_t> shape;
        indexArray material_indices = m_baseIO->m_kernel->template loadArray<indexArray::element_type>(
            *desc.dataRoot,
            *desc.data,
            shape
        );

        IndexChannelPtr material_indices_channel = std::make_shared<IndexChannel>(shape[0], shape[1], material_indices);

        mesh->addIndexChannel(material_indices_channel, "cluster_material_indices");
    }

    
}

} // namespace meshio
} // namespace lvr2
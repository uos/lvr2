#include "MeshIO.hpp"
#include <iomanip>
#include <lvr2/util/Progress.hpp>
#include <lvr2/io/meshio/yaml/Texture.hpp>
#include <lvr2/io/meshio/yaml/Material.hpp>


namespace lvr2
{

template <typename FeatureBase>
void MeshIO<FeatureBase>::saveMesh(
    std::string mesh_name, 
    MeshBufferPtr mesh
    ) const
{
    std::cout << timestamp << "[MeshIO] Saving vertices" << std::endl;

    // Step 1: Save vertices
    auto desc = m_featureBase->m_schema->vertexChannel(mesh_name, "coordinates");
    // Write the vertices
    m_featureBase->m_kernel->saveFloatArray(
        *desc.dataRoot,
        *desc.data,
        {mesh->numVertices(), 3},
        mesh->getVertices());

    {
        YAML::Node meta;
        meta["data_type"] = "float";
        meta["entity"] = "channel";
        meta["type"] = "array";
        meta["name"] = "indices";
        meta["shape"].push_back(mesh->numVertices());
        meta["shape"].push_back(3);
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            meta
        );
    }

    // Step 2: Save cluster/surfaces
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

    // Get highest cluster number to use in progress bar
    size_t n_cluster = 0;
    while(true)
    {
        std::string cluster_name = "cluster" + std::to_string(n_cluster) + "_face_indices";
        if (!mesh->getIndexChannel(cluster_name)) break;
        n_cluster++;
    }

    // display progress bar
    ProgressBar surface_progress(n_cluster, timestamp.getElapsedTime() + "[MeshIO] Saving surfaces");

    size_t cluster_idx = 0;
    while(true)
    {
        ////            ////
        //  Prepare Data  //
        ////            ////

        // Naming structure see FinalizeAlgorithms.tcc TextureFinalizer::apply
        std::string cluster_name = "cluster" + std::to_string(cluster_idx) + "_face_indices";
        // Contains all face indices of the current cluster
        IndexChannelOptional    clf_idx_opt = mesh->getIndexChannel(cluster_name);

        // If no channel with index surface_idx is found end loop
        if (!clf_idx_opt) break;

        // continuous index buffer for the clusters faces
        std::vector<IndexChannelOptional::value_type::DataType> idx_buffer;
        // continuous index buffer for the text coords
        std::vector<float> uv_buffer;
        

        // Add all face indices to continuous buffer
        // clfi is an index into the clusters face array
        for (size_t clf_i = 0; clf_i < clf_idx_opt->numElements(); clf_i++)
        {
            auto face_idx = (*clf_idx_opt)[clf_i];
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
        

        cluster_idx++;
        ++surface_progress;
    }
    std::cout << std::endl;


    // Step 3: Save all Materials
    const auto& materials = mesh->getMaterials();
    ProgressBar material_progress( materials.size(), timestamp.getElapsedTime() + "[MeshIO] Saving materials & textures");
    for (size_t idx = 0; idx < materials.size(); idx++)
    {
        m_materialIO->saveMaterial(
            mesh_name,
            idx,
            mesh
        );
        ++material_progress;
    }
    std::cout << std::endl;
}

} // namespace lvr2
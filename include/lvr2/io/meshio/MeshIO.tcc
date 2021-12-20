#include "MeshIO.hpp"
#include <iomanip>
#include <lvr2/util/Progress.hpp>
#include <lvr2/io/meshio/yaml/Texture.hpp>
#include <lvr2/io/meshio/yaml/Material.hpp>


namespace lvr2
{

template <typename FeatureBase>
void MeshIO<FeatureBase>::saveMesh(
    const std::string mesh_name, 
    const MeshBufferPtr mesh
    ) const
{
    std::cout << timestamp << "[MeshIO] Saving vertices" << std::endl;
    std::cout << timestamp << "[MeshIO] ===== ADD SUPPORT FOR OTHER CHANNELS! =====" << std::endl;

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
        // Naming structure see FinalizeAlgorithms.tcc TextureFinalizer::apply
        std::string cluster_name = "cluster" + std::to_string(cluster_idx) + "_face_indices";
        // Contains all face indices of the current cluster
        IndexChannelOptional    clf_idx_opt = mesh->getIndexChannel(cluster_name);

        // If no channel with index surface_idx is found end loop
        if (!clf_idx_opt) break;

        // Save the cluster
        m_clusterIO->saveCluster(
            mesh_name,
            cluster_idx,
            mesh,
            *clf_idx_opt
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
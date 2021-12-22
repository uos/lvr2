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

    // Step 0: Write Mesh meta TODO: wrtie YAML conversion
    {
        Description desc = m_featureBase->m_schema->mesh(mesh_name);
        YAML::Node node;
        node["n_materials"] = (uint64_t) mesh->getMaterials().size();
        node["n_textures"] = (uint64_t) mesh->getTextures().size();
        node["n_faces"] = (uint64_t) mesh->numFaces();
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }

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
        meta["shape"] = YAML::Load("[]");
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
    // Write data
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

template <typename FeatureBase>
MeshBufferPtr MeshIO<FeatureBase>::loadMesh(const std::string& name) const
{

    auto ret = MeshBufferPtr(new MeshBuffer); // Buffer for the loaded mesh 
    floatArr vertices; // Buffer for the loaded vertices

    Description desc = m_featureBase->m_schema->mesh(name);

    // Check if the mesh exists
    if (!m_featureBase->m_kernel->exists(*desc.dataRoot))
    {
        std::cout << timestamp << "[MeshIO] Mesh '" << name << "' not found." << std::endl;
        return nullptr;
    }


    desc = m_featureBase->m_schema->vertexChannel(name, "coordinates");
    // Load vertex meta
    YAML::Node node;
    m_featureBase->m_kernel->loadMetaYAML(
        *desc.metaRoot,
        *desc.meta,
        node
    );

    // Load vertices
    auto dims =  node["shape"].as<std::vector<size_t>>();

    vertices = m_featureBase->m_kernel->loadFloatArray(
        *desc.dataRoot,
        *desc.data,
        dims
    );
    // setVertices takes ownership of the array; std::move is needed because it expects an rvalue&
    ret->setVertices(std::move(vertices), dims[0]);
    ret->setTextureCoordinates(floatArr(new float[dims[0] * 2])); // Allocate space for texture coordinates
    // TODO: Support other channels

    // === Cluster === //
    std::vector<indexArray::element_type> faces;
    std::vector<indexArray::element_type> cluster_materials;

    // Load all surfaces/cluster
    size_t cluster_idx = 0;
    while(true)
    {
        auto cluster = m_featureBase->m_clusterIO->loadCluster(
            name,
            cluster_idx,
            ret->getTextureCoordinates(),
            faces
        );

        if (!cluster) break;

        // Insert cluster to mesh
        std::string cluster_name = "cluster" + std::to_string(cluster_idx) + "_face_indices";
        ret->addIndexChannel(
            cluster->face_indices, 
            cluster_name, 
            cluster->num_faces, 
            1);
        
        cluster_materials.push_back(cluster->material_index);

        cluster_idx++;
    }
    // Insert cluster -> material map
    ret->addIndexChannel(
        Util::convert_vector_to_shared_array(cluster_materials),
        "cluster_material_indices",
        cluster_materials.size(),
         1);
    // Add faces to mesh
    ret->setFaceIndices(Util::convert_vector_to_shared_array(faces), faces.size() / 3);


    // Load num_materials and num_textures to allocate memory
    {
        Description desc = m_featureBase->m_schema->mesh(name);
        YAML::Node node;
        m_featureBase->m_kernel->loadMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
        ret->getMaterials().reserve(node["n_materials"].as<uint64_t>()); // Reserve memory
        ret->getTextures().resize(node["n_textures"].as<uint64_t>()); // Resize for index based access
    }

    
    size_t material_idx = 0;
    while(true)
    {
        auto res = m_featureBase->m_materialIO->loadMaterial(
            name,
            material_idx
        );

        if (!res.first)
        {
            break;
        }

        ret->getMaterials().push_back(*res.first);
        
        if (res.second)
        {
            ret->getTextures()[res.second->m_index] = *res.second;
        }

        material_idx++;
    }



    // Last TODO: Reconstruct Face array, reconstruct texture_coordinate array
    

    return ret;
}

} // namespace lvr2
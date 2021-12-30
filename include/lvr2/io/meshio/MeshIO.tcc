#include "MeshIO.hpp"
#include <iomanip>
#include <lvr2/util/Progress.hpp>
#include <lvr2/io/meshio/yaml/Texture.hpp>
#include <lvr2/io/meshio/yaml/Material.hpp>
#include <lvr2/io/meshio/yaml/ArrayMeta.hpp>
#include <lvr2/io/meshio/ArrayMeta.hpp>


namespace lvr2
{

template <typename FeatureBase>
void MeshIO<FeatureBase>::saveMesh(
    const std::string mesh_name, 
    const MeshBufferPtr mesh
    ) const
{
    std::cout << timestamp << "[MeshIO] Saving '" << mesh_name << "' to " 
        << m_featureBase->m_kernel->fileResource() << std::endl;
    // Step 0: Write Mesh meta 
    // TODO: write YAML conversion
    {
        Description desc = m_featureBase->m_schema->mesh(mesh_name);
        YAML::Node node;
        node["n_materials"] = (uint64_t) mesh->getMaterials().size();
        node["n_textures"] = (uint64_t) mesh->getTextures().size();
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }

    // Step 1: Save vertices
    saveVertices(mesh_name, mesh);

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
    m_featureBase->m_materialIO->saveMaterials(
        mesh_name,
        mesh
    );
}

template <typename FeatureBase>
MeshBufferPtr MeshIO<FeatureBase>::loadMesh(const std::string& name) const
{

    auto ret = MeshBufferPtr(new MeshBuffer); // Buffer for the loaded mesh 
    Description desc = m_featureBase->m_schema->mesh(name);

    // Check if the mesh exists
    if (!m_featureBase->m_kernel->exists(*desc.dataRoot))
    {
        std::cout << timestamp << "[MeshIO] Mesh '" << name << "' not found." << std::endl;
        return nullptr;
    }
    std::cout << timestamp << "[MeshIO] Loading '" << name << "' from " 
        << m_featureBase->m_kernel->fileResource() << std::endl;

    // === Vertices === //
    loadVertices(name, ret);
    
    // === Cluster === //
    size_t n_clusters = loadSurfaces(name, ret);

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

    // === Materials and Textures === //
    size_t n_materials = m_featureBase->m_materialIO->loadMaterials(name, ret);

    return ret;
}


template <typename FeatureBase>
size_t MeshIO<FeatureBase>::loadSurfaces(
    const std::string& mesh_name,
    MeshBufferPtr mesh) const
{
    std::vector<indexArray::element_type> faces;
    std::vector<indexArray::element_type> faceToMaterial;
    std::vector<indexArray::element_type> cluster_materials;

    // Count clusters in file
    size_t count = 0;
    while(true)
    {
        Description desc = m_featureBase->m_schema->surface(mesh_name, count);
        if (!m_featureBase->m_kernel->exists(
            *desc.dataRoot,
            *desc.data
        )) break;
        count++;
    }

    ProgressBar bar(count, timestamp.getElapsedTime() + "[MeshIO] Loading surfaces");
    // Load all surfaces/cluster
    size_t cluster_idx = 0;
    while(true)
    {
        auto cluster = m_featureBase->m_clusterIO->loadCluster(
            mesh_name,
            cluster_idx,
            mesh->getTextureCoordinates(),
            faces,
            faceToMaterial
        );

        if (!cluster) break;

        // Insert cluster to mesh
        std::string cluster_name = "cluster" + std::to_string(cluster_idx) + "_face_indices";
        mesh->addIndexChannel(
            cluster->face_indices, 
            cluster_name, 
            cluster->num_faces, 
            1);
        
        cluster_materials.push_back(cluster->material_index);

        ++bar;
        ++cluster_idx;
    }
    std::cout << std::endl;
    // Insert cluster -> material map
    mesh->addIndexChannel(
        Util::convert_vector_to_shared_array(cluster_materials),
        "cluster_material_indices",
        cluster_materials.size(),
         1);
    // Add faces to mesh
    mesh->setFaceIndices(Util::convert_vector_to_shared_array(faces), faces.size() / 3);
    // Insert face -> material map
    mesh->setFaceMaterialIndices(
        Util::convert_vector_to_shared_array(faceToMaterial)
        );

    return cluster_idx;
}

template <typename FeatureBase>
void MeshIO<FeatureBase>::saveVertices(std::string mesh_name, MeshBufferPtr mesh) const
{
    std::cout << timestamp << "[MeshIO] Mesh has vertex coordinates: " << (mesh->hasVertices() ? "yes" : "no") << std::endl;
    std::cout << timestamp << "[MeshIO] Mesh has vertex normals:     " << (mesh->hasVertexNormals() ? "yes" : "no") << std::endl;
    std::cout << timestamp << "[MeshIO] Mesh has vertex colors:      " << (mesh->hasVertexColors() ? "yes" : "no") << std::endl;

    if (mesh->hasVertices())
    {
        auto desc = m_featureBase->m_schema->vertexChannel(mesh_name, "coordinates");

        // Write the vertices
        m_featureBase->m_kernel->saveFloatArray(
        *desc.dataRoot,
        *desc.data,
        {mesh->numVertices(), 3},
        mesh->getVertices());

        meshio::ArrayMeta meta;
        meta.data_type = "float",
        meta.shape = {mesh->numVertices(), 3};

        YAML::Node node;
        node = meta;
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }



    if (mesh->hasVertexNormals())
    {
        auto desc = m_featureBase->m_schema->vertexChannel(mesh_name, "normals");

        // Write the vertices
        m_featureBase->m_kernel->saveFloatArray(
        *desc.dataRoot,
        *desc.data,
        {mesh->numVertices(), 3},
        mesh->getVertexNormals());
    
        meshio::ArrayMeta meta;
        meta.data_type = "float",
        meta.shape = {mesh->numVertices(), 3};

        YAML::Node node;
        node = meta;
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }



    if (mesh->hasVertexColors())
    {
        auto desc = m_featureBase->m_schema->vertexChannel(mesh_name, "colors");
        size_t color_w;
        auto colors = mesh->getVertexColors(color_w);
        // Write the vertices
        m_featureBase->m_kernel->saveUCharArray(
        *desc.dataRoot,
        *desc.data,
        {mesh->numVertices(), color_w},
        colors);
    
        meshio::ArrayMeta meta;
        meta.data_type = "uchar",
        meta.shape = {mesh->numVertices(), color_w};

        YAML::Node node;
        node = meta;
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }

}

template <typename FeatureBase>
void MeshIO<FeatureBase>::loadVertices(std::string mesh_name, MeshBufferPtr mesh) const
{
    // Check which channels exist
    auto coord_desc = m_featureBase->m_schema->vertexChannel(mesh_name, "coordinates");
    auto normal_desc = m_featureBase->m_schema->vertexChannel(mesh_name, "normals");
    auto color_desc = m_featureBase->m_schema->vertexChannel(mesh_name, "colors");

    bool hasCoords  = false;
    bool hasNormals = false;
    bool hasColors  = false;

    if (m_featureBase->m_kernel->exists(*coord_desc.dataRoot, *coord_desc.data)) hasCoords     = true;
    if (m_featureBase->m_kernel->exists(*normal_desc.dataRoot, *normal_desc.data)) hasNormals    = true;
    if (m_featureBase->m_kernel->exists(*color_desc.dataRoot, *color_desc.data)) hasColors     = true;

    std::cout << timestamp << "[MeshIO] Mesh has vertex coordinates: " << (hasCoords  ? "yes" : "no") << "\n";
    std::cout << timestamp << "[MeshIO] Mesh has vertex normals:     " << (hasNormals ? "yes" : "no") << "\n";
    std::cout << timestamp << "[MeshIO] Mesh has vertex colors:      " << (hasColors  ? "yes" : "no") << "\n";
    std::cout << std::flush;


    // === Vertex Coordinates === //
    if (hasCoords)
    {
        // Load vertex meta
        YAML::Node node;
        m_featureBase->m_kernel->loadMetaYAML(
            *coord_desc.metaRoot,
            *coord_desc.meta,
            node
        );
        meshio::ArrayMeta meta = node.as<meshio::ArrayMeta>();
        // Check data type
        if (meta.data_type == "float")
        {
            // Load vertices
            auto vertices = m_featureBase->m_kernel->loadFloatArray(
                *coord_desc.dataRoot,
                *coord_desc.data,
                meta.shape
            );
            // setVertices takes ownership of the array; std::move is needed because it expects an rvalue&
            mesh->setVertices(std::move(vertices), meta.shape[0]);
            mesh->setTextureCoordinates(floatArr(new float[meta.shape[0] * 2])); // Allocate space for texture coordinates
        }
        else
        {
            std::cout << timestamp << "[MeshIO] Array 'coordinates' data type '" << meta.data_type << "' is not 'float'" << std::endl; 
        }
    }


    // === Vertex Normals === //
    if (hasNormals)
    {
        // Load vertex meta
        YAML::Node node;
        m_featureBase->m_kernel->loadMetaYAML(
            *normal_desc.metaRoot,
            *normal_desc.meta,
            node
        );
        meshio::ArrayMeta meta = node.as<meshio::ArrayMeta>();
        // Check data type
        if (meta.data_type == "float")
        {
            // Load vertices
            auto normals = m_featureBase->m_kernel->loadFloatArray(
                *normal_desc.dataRoot,
                *normal_desc.data,
                meta.shape
            );
            mesh->setVertexNormals(std::move(normals));
        }
        else
        {
            std::cout << timestamp << "[MeshIO] Array 'normals' data type '" << meta.data_type << "' is not 'float'" << std::endl; 
        }
    }
    

    // === Vertex Colors === //
    if (hasColors)
    {
        YAML::Node node;
        // Load color meta
        m_featureBase->m_kernel->loadMetaYAML(
            *color_desc.metaRoot,
            *color_desc.meta,
            node
        );
        meshio::ArrayMeta meta = node.as<meshio::ArrayMeta>();
        // Check data type
        if (meta.data_type == "uchar")
        {
            // Load colors
            auto colors = m_featureBase->m_kernel->loadUCharArray(
            *color_desc.dataRoot,
            *color_desc.data,
            meta.shape
            );
            mesh->setVertexColors(std::move(colors), meta.shape[1]);
        }
        else
        {
            std::cout << timestamp << "[MeshIO] Array 'colors' data type '" << meta.data_type << "' is not 'uchar'" << std::endl; 
        }
    }
    
}

} // namespace lvr2

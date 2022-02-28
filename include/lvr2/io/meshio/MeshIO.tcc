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
        Description desc = m_featureBase->m_description->mesh(mesh_name);
        YAML::Node node;
        node["n_materials"] = (uint64_t) mesh->getMaterials().size();
        node["n_textures"]  = (uint64_t) mesh->getTextures().size();
        node["n_faces"]     = (uint64_t) mesh->numFaces();
        m_featureBase->m_kernel->saveMetaYAML(
            *desc.metaRoot,
            *desc.meta,
            node
        );
    }

    // Step 1: Save vertices
    saveVertices(mesh_name, mesh);

    // Save faces
    m_faceIO->saveFaces(mesh_name, mesh);

    // Save clusters
    m_clusterIO->saveClusters(mesh_name, mesh);


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
    Description desc = m_featureBase->m_description->mesh(name);

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
    
    // === Faces === //
    m_faceIO->loadFaces(name, ret);

    // === Cluster === //
    m_clusterIO->loadClusters(name, ret);

    // Load num_materials and num_textures to allocate memory
    {
        Description desc = m_featureBase->m_description->mesh(name);
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
void MeshIO<FeatureBase>::saveVertices(std::string mesh_name, MeshBufferPtr mesh) const
{
    std::cout << timestamp << "[MeshIO] Mesh has vertex coordinates:  " << (mesh->hasVertices() ? "yes" : "no") << std::endl;
    std::cout << timestamp << "[MeshIO] Mesh has vertex normals:      " << (mesh->hasVertexNormals() ? "yes" : "no") << std::endl;
    std::cout << timestamp << "[MeshIO] Mesh has vertex colors:       " << (mesh->hasVertexColors() ? "yes" : "no") << std::endl;

    if (mesh->hasVertices())
    {
        auto desc = m_featureBase->m_description->vertexChannel(mesh_name, "coordinates");

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
        auto desc = m_featureBase->m_description->vertexChannel(mesh_name, "normals");

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
        auto desc = m_featureBase->m_description->vertexChannel(mesh_name, "colors");
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

    if (mesh->getTextureCoordinates())
    {
        auto desc = m_featureBase->m_description->vertexChannel(mesh_name, "texture_coordinates");
        // Write the vertices
        m_featureBase->m_kernel->saveFloatArray(
        *desc.dataRoot,
        *desc.data,
        {mesh->numVertices(), 2},
        mesh->getTextureCoordinates());
    
        meshio::ArrayMeta meta;
        meta.data_type = "float",
        meta.shape = {mesh->numVertices(), 2};

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
    auto coord_desc = m_featureBase->m_description->vertexChannel(mesh_name, "coordinates");
    auto normal_desc = m_featureBase->m_description->vertexChannel(mesh_name, "normals");
    auto color_desc = m_featureBase->m_description->vertexChannel(mesh_name, "colors");
    auto tex_coord_desc = m_featureBase->m_description->vertexChannel(mesh_name, "texture_coordinates");

    bool hasCoords      = false;
    bool hasNormals     = false;
    bool hasColors      = false;
    bool hasTexCoords   = false;

    if (m_featureBase->m_kernel->exists(*coord_desc.dataRoot, *coord_desc.data)) hasCoords     = true;
    if (m_featureBase->m_kernel->exists(*normal_desc.dataRoot, *normal_desc.data)) hasNormals    = true;
    if (m_featureBase->m_kernel->exists(*color_desc.dataRoot, *color_desc.data)) hasColors     = true;
    if (m_featureBase->m_kernel->exists(*tex_coord_desc.dataRoot, *tex_coord_desc.data)) hasTexCoords     = true;

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

    // === Texture Coordinates === //
    if (hasTexCoords)
    {
        YAML::Node node;
        // Load meta
        m_featureBase->m_kernel->loadMetaYAML(
            *tex_coord_desc.metaRoot,
            *tex_coord_desc.meta,
            node
        );
        meshio::ArrayMeta meta = node.as<meshio::ArrayMeta>();
        if (meta.data_type == "float")
        {
            // Load texture coordinates
            auto coords = m_featureBase->m_kernel->loadFloatArray(
                *tex_coord_desc.dataRoot,
                *tex_coord_desc.data,
                meta.shape
            );
            mesh->setTextureCoordinates(std::move(coords));
        }
        else
        {
            std::cout << timestamp << "[MeshIO] Array 'texture_coordinates' data type '" << meta.data_type << "' is not 'float'" << std::endl; 
        }
    }
    
}

} // namespace lvr2

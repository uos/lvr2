#include MeshIO.hpp

/*

Structure:
    MeshBufferPtr               // Mesh
    ClusterBiMap                // Clusters
    MaterializerResult          // Materials and Textures
        m_clusterMaterials      // Material for each cluster handle
        m_textures              // The textures generated for each cluster

*/

namespace lvr2
{

template <typename ClusterHandleT, typename MaterialHandleT>
void MeshIO::saveMesh(
    std::string mesh_name, 
    MeshBufferPtr mesh, 
    ClusterBiMap<ClusterHandleT> clusters,
    MaterializerResult<MaterialHandleT> materials)
{
    // Step 1: Save vertices
    auto desc = m_schema->vertexChannel(mesh_name, "coordinates");
    // Write the vertices
    m_kernel->saveFloatArray(
        desc.dataRoot,
        desc.data,
        {mesh->numVertices(), 3},
        mesh->getVertices());

    // Step 2: Save cluster/surfaces


    // Step 3/2.5: Save Materials
}

} // namespace lvr2
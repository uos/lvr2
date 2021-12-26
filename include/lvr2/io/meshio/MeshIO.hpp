#pragma once

#include <lvr2/io/meshio/FeatureBase.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MaterialIO.hpp>
#include <lvr2/io/meshio/ClusterIO.hpp>

namespace lvr2
{

template <typename FeatureBase>
class MeshIO
{
public:
    void saveMesh(
        const std::string mesh_name, 
        const MeshBufferPtr mesh
        ) const;
    
    MeshBufferPtr loadMesh(
        const std::string& mesh_name
    ) const;

private:
    /**
     * @brief Saves the vertices, normals and colors if available
     * 
     * @param mesh_name The name of the mesh
     * @param mesh The mesh from which to save the data
     */
    void saveVertices(std::string mesh_name, MeshBufferPtr mesh) const;

    /**
     * @brief Loads vertices, normals and colors if available
     * 
     * @param mesh_name The name of the mesh to load
     * @param[out] mesh The mesh to add the vertices, normals and colors
     */
    void loadVertices(std::string mesh_name, MeshBufferPtr mesh) const;

    /**
     * @brief Loads all Materials associated with \p mesh_name
     * 
     * @param mesh_name The name of the Mesh in the h5 file
     * @param[out] mesh The Materials and textures will be added to this mesh. \p mesh needs to have enough memory allocated to hold all textures
     * @return The number of Materials loaded
     */
    size_t loadMaterials(const std::string& mesh_name, MeshBufferPtr mesh) const;
    /**
     * @brief Loads all clusters associated with \p mesh_name
     * 
     * @param mesh_name The name of the Mesh in the h5 file
     * @param[out] mesh The surfaces will be added to this Mesh 
     * @return size_t The number of surfaces read
     */
    size_t loadSurfaces(const std::string& mesh_name, MeshBufferPtr mesh) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    MaterialIO<FeatureBase>* m_materialIO 
        = static_cast<MaterialIO<FeatureBase>*>(m_featureBase);

    ClusterIO<FeatureBase>* m_clusterIO 
        = static_cast<ClusterIO<FeatureBase>*>(m_featureBase);

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<MeshIO, FeatureBase>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<MaterialIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<ClusterIO, FeatureBase>::type;

    using deps = typename dep1::template Merge<dep2>;

    // Add the feature
    using type = typename deps::template add_features<MeshIO>::type;
};

} // namespace lvr2

#include "MeshIO.tcc"
#pragma once

#include <lvr2/io/baseio/BaseIO.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/meshio/MaterialIO.hpp>
#include <lvr2/io/meshio/ClusterIO.hpp>
#include <lvr2/io/meshio/FaceIO.hpp>

namespace lvr2
{
namespace meshio
{

template <typename BaseIO>
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

protected:

    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    MaterialIO<BaseIO>* m_materialIO = static_cast<MaterialIO<BaseIO>*>(m_baseIO);
    ClusterIO<BaseIO>* m_clusterIO = static_cast<ClusterIO<BaseIO>*>(m_baseIO);
    FaceIO<BaseIO>* m_faceIO = static_cast<FaceIO<BaseIO>*>(m_baseIO);

};
} // namespace meshio 

template <typename FB>
struct FeatureConstruct<lvr2::meshio::MeshIO, FB>
{
    // Dependencies
    using dep1 = typename FeatureConstruct<lvr2::meshio::MaterialIO, FB>::type;
    using dep2 = typename FeatureConstruct<lvr2::meshio::ClusterIO, FB>::type;
    using dep3 = typename FeatureConstruct<lvr2::meshio::FaceIO, FB>::type;

            using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

            // Add the feature
            using type = typename deps::template add_features<lvr2::meshio::MeshIO>::type;
};


} // namespace lvr2

#include "MeshIO.tcc"
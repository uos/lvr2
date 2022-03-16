#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/schema/MeshSchema.hpp>
#include <lvr2/io/DataStruct.hpp>

namespace lvr2
{
namespace meshio
{

struct ClusterIOData
{
    size_t num_faces = 0;
    size_t material_index = 0;
    indexArray face_indices = nullptr;
};

template <typename BaseIO>
class ClusterIO
{
public:

    void saveClusters(
        const std::string& mesh_name,
        const MeshBufferPtr mesh
    );

    /**
     * @brief
     *
     * @param mesh_name
     * @param[out] mesh The MeshBuffer the data is written to
     */
    void loadClusters(
        const std::string& mesh_name,
        MeshBufferPtr mesh
    );

protected:
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
};

} // namespace meshio
} // namespace lvr2

#include "ClusterIO.tcc"
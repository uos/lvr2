#pragma once
#include "lvr2/io/MeshBuffer.hpp"

namespace lvr2
{

template <typename FeatureBase>
class FaceIO
{
public:
    /**
     * @brief Saves all faces and corresponding data
     * 
     * @param mesh_name The mesh to save to
     * @param mesh The MeshBuffer to take the data from
     */
    void saveFaces(
        const std::string& mesh_name,
        const MeshBufferPtr mesh
    ) const;

    /**
     * @brief Loads all faces and corresponding data
     * 
     * @param mesh_name The mesh from which to load the data
     * @param[out] mesh The MeshBuffer to add the data to
     */
    void loadFaces(
        const std::string& mesh_name,
        MeshBufferPtr mesh
    ) const;
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

private:

    bool saveFaceIndices(
        const std::string& mesh_name,
        const MeshBufferPtr mesh) const;
    
    bool saveFaceColors(
        const std::string& mesh_name,
        const MeshBufferPtr mesh) const;
    
    bool saveFaceNormals(
        const std::string& mesh_name,
        const MeshBufferPtr mesh) const;

    bool saveFaceMaterialIndices(
        const std::string& mesh_name,
        const MeshBufferPtr mesh) const;

    bool loadFaceIndices(
        const std::string& mesh_name,
        MeshBufferPtr mesh) const;
    
    bool loadFaceColors(
        const std::string& mesh_name,
        MeshBufferPtr mesh) const;
    
    bool loadFaceNormals(
        const std::string& mesh_name,
        MeshBufferPtr mesh) const;

    bool loadFaceMaterialIndices(
        const std::string& mesh_name,
        MeshBufferPtr mesh) const;
};

} // namespace lvr2

#include "FaceIO.tcc"
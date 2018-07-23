/* Copyright (C) 2011 Uni Osnabrück
* This file is part of the LAS VEGAS Reconstruction Toolkit,
*
* LAS VEGAS is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* LAS VEGAS is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
*/

/*
* Materializer.hpp
*
*  @date 17.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#ifndef LVR2_ALGORITHM_MATERIALIZER_H_
#define LVR2_ALGORITHM_MATERIALIZER_H_

#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/optional.hpp>

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/texture/ClusterTexCoordMapping.hpp>
#include <lvr2/texture/Texture.hpp>
#include <lvr2/util/ClusterBiMap.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/geometry/BoundingRectangle.hpp>
#include <lvr2/algorithm/Texturizer.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>


#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>
#include <unordered_map>

namespace lvr2
{

/**
 * @struct MaterializerResult
 * @brief Result struct for the materializer
 *
 * This struct is a wrapper type for the result of the materializer. It contains a map from cluster to material and may
 * contain optional texture data
 */
template<typename BaseVecT>
struct MaterializerResult
{
    /// Materials for each cluster
    DenseClusterMap<Material> m_clusterMaterials;

    /// A stable vector of textures. each texture is identified by a tex.-handle
    optional<StableVector<TextureHandle, Texture<BaseVecT>>> m_textures;

    /// Cluster texture coordinates for each vertex
    optional<SparseVertexMap<ClusterTexCoordMapping>> m_vertexTexCoords;

    /// Keypoints
    optional<std::unordered_map<BaseVecT, std::vector<float>>> m_keypoints;

    /**
     * @brief Constructor
     *
     * @param clusterMaterials The cluster material map
     */
    MaterializerResult(
        DenseClusterMap<Material> clusterMaterials
    )   :
        m_clusterMaterials(clusterMaterials)
    {
    }

    /**
     * @brief Constructor
     *
     * @param clusterMaterials The cluster material map
     * @param textures The stable vector of textures
     * @param vertexTexCoords The vertex texture coordinates
     * @param keypoints The keypoints
     */
    MaterializerResult(
        DenseClusterMap<Material> clusterMaterials,
        StableVector<TextureHandle, Texture<BaseVecT>> textures,
        SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords,
        std::unordered_map<BaseVecT, std::vector<float>> keypoints
    ) :
        m_clusterMaterials(clusterMaterials),
        m_textures(textures),
        m_vertexTexCoords(vertexTexCoords),
        m_keypoints(keypoints)
    {
    }

};

/**
 * @class Materializer
 * @brief Class for calculating materials for each cluster of a given mesh
 *
 * For each cluster of a given mesh, this class generates a material. Based on the original pointcloud, it calculates a
 * color for each cluster. If a Texturizer is provided, it generates a texture for the material instead. In that
 * case it will also calculate texture coordinates and keypoints.
 */
template<typename BaseVecT>
class Materializer
{

public:

    /**
     * @brief Constructor
     *
     * @param mesh The mesh
     * @param cluster The cluster map
     * @param normals The normals map
     * @param surface The point cloud
     */
    Materializer(
        const BaseMesh<BaseVecT>& mesh,
        const ClusterBiMap<FaceHandle>& cluster,
        const FaceMap<Normal<BaseVecT>>& normals,
        const PointsetSurface<BaseVecT>& surface
    );

    /**
     * @brief Sets the texturizer
     * @param texturizer The texturizer
     */
    void setTexturizer(Texturizer<BaseVecT> texturizer);

    /**
     * @brief Generates materials
     *
     * For each cluster, check whether a texture should be generated:
     * - If no texturizer is provided, no textures will be generated.
     * - If a texturizer is provided, count the number of faces in that cluster and generate a texture if the number of
     *   faces is within the defined interval (which is stored in the texturizer).
     *
     * If no texture should be generated, calculate the median color of the faces in the cluster. Then create a material
     * using this color if it doesn't already exist.
     *
     * If textures should be generated, calculate the contour vertices of the cluster and a bounding rectangle and let
     * the texturizer generate a texture using the bounding rectangle.
     * Then calculate AKAZE keypoints for the texture image and texture coordinates for each vertex in the cluster.
     *
     * @return The materializer result, that contains materials and optional texture data
     */
    MaterializerResult<BaseVecT> generateMaterials();

    /**
     * @brief Saves the textures by calling the `saveTextures()` method of the texturizer
     */
    void saveTextures();

private:

    /// Mesh
    const BaseMesh<BaseVecT>& m_mesh;
    /// Clusters
    const ClusterBiMap<FaceHandle>& m_cluster;
    /// Normals
    const FaceMap<Normal<BaseVecT>>& m_normals;
    /// Point cloud
    const PointsetSurface<BaseVecT>& m_surface;

    /// Texturizer
    optional<Texturizer<BaseVecT>&> m_texturizer;

};

} // namespace lvr2

#include <lvr2/algorithm/Materializer.tcc>

#endif /* LVR2_ALGORITHM_MATERIALIZER_H_ */

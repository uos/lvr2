/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Handles.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/texture/ClusterTexCoordMapping.hpp"
#include "lvr2/texture/Texture.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/texture/Material.hpp"
#include "lvr2/geometry/BoundingRectangle.hpp"
#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"


#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"
#include <unordered_map>
#include <unordered_set>

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
    boost::optional<StableVector<TextureHandle, Texture>> m_textures;

    /// Cluster texture coordinates for each vertex
    boost::optional<SparseVertexMap<ClusterTexCoordMapping>> m_vertexTexCoords;

    /// Keypoints
    boost::optional<std::unordered_map<BaseVecT, std::vector<float>>> m_keypoints;

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
        StableVector<TextureHandle, Texture> textures,
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
        const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
        const PointsetSurface<BaseVecT>& surface
    );

    /**
     * @brief Sets the texturizer
     * @param texturizer The texturizer
     */
    void setTexturizer(Texturizer<BaseVecT>& texturizer);

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
    const FaceMap<Normal<typename BaseVecT::CoordType>>& m_normals;
    /// Point cloud
    const PointsetSurface<BaseVecT>& m_surface;

    /// Texturizer
    boost::optional<Texturizer<BaseVecT>&> m_texturizer;

};

} // namespace lvr2

#include "lvr2/algorithm/Materializer.tcc"

#endif /* LVR2_ALGORITHM_MATERIALIZER_H_ */

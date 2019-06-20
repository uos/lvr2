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
* Texturizer.hpp
*
*  @date 15.09.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#ifndef LVR2_ALGORITHM_TEXTURIZER_H_
#define LVR2_ALGORITHM_TEXTURIZER_H_

#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Handles.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/geometry/BoundingRectangle.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/texture/ClusterTexCoordMapping.hpp"
#include "lvr2/texture/Texture.hpp"
#include "lvr2/texture/Material.hpp"
#include "lvr2/util/ClusterBiMap.hpp"

#include <opencv2/features2d.hpp>

namespace lvr2
{

/**
 * @class Texturizer
 * @brief Class that performs texture-related tasks
 */
template<typename BaseVecT>
class Texturizer
{

public:

    /**
     * @brief Constructor
     *
     * @param texelSize The size of one texture pixel, relative to the coordinate system of the point cloud
     * @param texMinClusterSize The minimum number of faces a cluster needs to be texturized
     * @param texMaxClusterSize The maximum number of faces a cluster needs to be texturized
     */
    Texturizer(
        float texelSize,
        int texMinClusterSize,
        int texMaxClusterSize
    );

    /**
     * @brief Get the texture to a given texture handle
     *
     * @param h The texture handle
     *
     * @return The texture
     */
    Texture getTexture(TextureHandle h);

    /**
     * @brief Returns all textures
     *
     * @return A StableVector containing all textures
     */
    StableVector<TextureHandle, Texture> getTextures();

    /**
     * @brief Get the texture index to a given texture handle
     *
     * @param h The texture handle
     *
     * @return The texture index
     */
    int getTextureIndex(TextureHandle h);

    /**
     * @brief Discover keypoints in a texture
     *
     * @param[in] texH Texture handle
     * @param[in] boundingRect Bounding rectangle computed for the texture
     * @param[in] detector Feature detector to use (any of @c cv::Feature2D)
     * @param[out] keypoints Vector of keypoints
     * @param[out] descriptors Matrix of descriptors for the keypoint
     */
    void findKeyPointsInTexture(const TextureHandle texH,
            const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
            const cv::Ptr<cv::Feature2D>& detector,
            std::vector<cv::KeyPoint>&
            keypoints, cv::Mat& descriptors);

    /**
     * @brief Compute 3D coordinates for texture-relative keypoints
     *
     * @param[in] keypoints Keypoints in image coordinates
     * @param[in] boundingRect Bounding rectangle of the texture embedded in 3D
     *
     * @return Vector of 3D coordinates of all keypoints
     */
    std::vector<BaseVecT> keypoints23d(const std::vector<cv::KeyPoint>&
        keypoints, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, const TextureHandle& h);

    /**
     * @brief Generates a texture for a given bounding rectangle
     *
     * Create a grid, based on given information (texel size, bounding rectangle).
     * For each cell in the grid (which represents a texel), let the `PointsetSurface` find the closest point in the
     * point cloud and use that point's color as color for the texel.
     *
     * @param index The index the texture will get
     * @param surface The point cloud
     * @param boundingRect The bounding rectangle of the cluster
     *
     * @return Texture handle of the generated texture
     */
    virtual TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
    );

    /**
     * @brief Calculate texture coordinates for a given 3D point in a texture
     *
     * @param texH The texture handle
     * @param boundingRect The bounding rectangle of the texture
     * @param v The 3D point
     *
     * @return The texture coordinates
     */
    TexCoords calculateTexCoords(
        TextureHandle texH,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
        BaseVecT v
    );

    /**
     * @brief Calculate a global 3D position for given texture coordinates
     *
     * @param texH The texture handle
     * @param boundingRect The bounding rectangle of the texture
     * @param coords The texture coordinates
     *
     * @return The 3D point
     */
    BaseVecT calculateTexCoordsInv(
        TextureHandle texH,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
        const TexCoords& coords
    );

    /**
     * @brief Calls the save method for each texture
     */
    void saveTextures();

    /// The size of a texture pixel
    const float m_texelSize;
    /// The minimum number of faces a cluster needs to be texturized
    const int m_texMinClusterSize;
    /// The maximum number of faces a cluster needs to be texturized
    const int m_texMaxClusterSize;

protected:

    /// StableVector, that contains all generated textures with texture handles
    StableVector<TextureHandle, Texture> m_textures;

};


} // namespace lvr2

#include "lvr2/algorithm/Texturizer.tcc"

#endif /* LVR2_ALGORITHM_TEXTURIZER_H_ */

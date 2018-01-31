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

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/texture/ClusterTexCoordMapping.hpp>
#include <lvr2/texture/Texture.hpp>
#include <lvr2/util/ClusterBiMap.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/geometry/BoundingRectangle.hpp>

#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>

#include <opencv2/features2d.hpp>

namespace lvr2
{

template<typename BaseVecT>
class Texturizer
{

public:

    Texturizer(
        float texelSize,
        int texMinClusterSize,
        int texMaxClusterSize
    );

    Texture<BaseVecT> getTexture(TextureHandle h);
    StableVector<TextureHandle, Texture<BaseVecT>> getTextures();
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
            const BoundingRectangle<BaseVecT>& boundingRect,
            const cv::Ptr<cv::Feature2D>& detector,
            std::vector<cv::KeyPoint>&
            keypoints, cv::Mat& descriptors);

    /**
     * @brief Compute 3D coordinates for texture-relative keypoints
     *
     * @param[in] keypoints Keypoints in image coordinates
     * @param[in] boundingRect Bounding rectangle of the texture embedded in 3D
     *
     * @return
     */
    std::vector<BaseVecT> keypoints23d(const std::vector<cv::KeyPoint>&
        keypoints, const BoundingRectangle<BaseVecT>& boundingRect, const TextureHandle& h);

    TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<BaseVecT>& boundingRect
    );

    TexCoords calculateTexCoords(
        TextureHandle texH,
        const BoundingRectangle<BaseVecT>& boundingRect,
        BaseVecT v
    );

    BaseVecT calculateTexCoordsInv(
        TextureHandle texH,
        const BoundingRectangle<BaseVecT>& boundingRect,
        const TexCoords& coords
    );

    void saveTextures();

    const float m_texelSize;
    const int m_texMinClusterSize;
    const int m_texMaxClusterSize;

private:

    StableVector<TextureHandle, Texture<BaseVecT>> m_textures;

};


} // namespace lvr2

#include <lvr2/algorithm/Texturizer.tcc>

#endif /* LVR2_ALGORITHM_TEXTURIZER_H_ */

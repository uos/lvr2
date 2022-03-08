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

#pragma once

// std includes
#include <memory>

// lvr2 includes
#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"
#include "lvr2/algorithm/raycasting/Intersection.hpp"
#include "lvr2/types/ScanTypes.hpp"

// Eigen
#include <Eigen/Dense>

namespace lvr2
{

template <typename BaseVecT>
class RayCastingTexturizer: public Texturizer<BaseVecT>
{
public:
    using Ptr = std::shared_ptr<RayCastingTexturizer<BaseVecT>>;
    using IntersectionT = Intersection<intelem::Face, intelem::Point, intelem::Distance>;
    
    RayCastingTexturizer() = delete;

    /**
     * @brief Construct a new Ray Casting Texturizer object
     * 
     * @param texelSize The size of one texture pixel, relative to the coordinate system of the point cloud
     * @param texMinClusterSize The minimum number of faces a cluster needs to be texturized
     * @param texMaxClusterSize The maximum number of faces a cluster needs to be texturized
     */
    RayCastingTexturizer(
        float texelMinSize,
        int texMinClusterSize,
        int texMaxClusterSize,
        const BaseMesh<BaseVector<float>>& geometry,
        const ClusterBiMap<FaceHandle>& clusters,
        const ScanProjectPtr project
    );

    /**
     * @brief Generates a texture for a given bounding rectangle
     *
     * Create a grid, based on given information (texel size, bounding rectangle).
     * For each cell in the grid (which represents a texel), cast a ray from the camera
     * to the texel in 3D space to check for visibility. If the texel is visible calculate
     * the texel color based on the RGB image data.
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
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
        ClusterHandle cluster
    ) override;

    void setGeometry(const BaseMesh<BaseVecT>& mesh);

    void setClusters(const ClusterBiMap<FaceHandle>& clusters);

    void setScanProject(const ScanProjectPtr project);

    virtual ~RayCastingTexturizer()
    {
        cv::namedWindow("debug", cv::WINDOW_NORMAL);
        int i = 0;
        for (auto& info: m_images)
        {
            if (info.image->loaded())
            {
                std::stringstream sstr;
                sstr << "cameraProjected" << i << ".bmp";
                cv::imwrite(sstr.str(), info.image->image);
            }
            i++;
        }
        
    }

private:
    template <typename... Args>
    Texture initTexture(Args&&... args) const;

    /**
     * @brief Calculates the uv coordinates for each pixel of the Texture
     * 
     * @param tex The Texture to calculate the uv coordinates for
     * @return std::vector<TexCoords> 
     */
    std::vector<TexCoords> calculateUVCoordsPerPixel(const Texture& tex) const;

    std::vector<Vector3f> calculate3DPointsPerPixel(const std::vector<TexCoords>&, const BoundingRectangle<typename BaseVecT::CoordType>&);

    std::vector<bool> calculateVisibilityPerPixel(
        const Vector3f from, 
        const std::vector<Vector3f>& to,
        const std::vector<bool>& texturized,
        const ClusterHandle cluster) const;

    void DEBUGDrawBorder(TextureHandle texH, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, ClusterHandle clusterH);

private:

    struct ImageInfo
    {
        Eigen::Quaternionf rotation;
        Eigen::Translation3f translation;
        CameraImagePtr image;
        PinholeModel model;
    };

    // The Raycaster which is used while raycasting
    RaycasterBasePtr<IntersectionT> m_tracer;

    // The clusters of faces
    ClusterBiMap<FaceHandle> m_clusters;

    // Maps the face indices given to embree to FaceHandles
    std::map<size_t, FaceHandle> m_embreeToHandle;

    // The images and poses used for texturization
    std::vector<ImageInfo> m_images;

    const BaseMesh<BaseVector<float>>& m_debug;
};

} // namespace lvr2

#include "RayCastingTexturizer.tcc"
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
 * ImageTexturizer.hpp
 *
 *  @date 01.11.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_ALGORITHM_IMAGETEXTURIZER_HPP
#define LVR2_ALGORITHM_IMAGETEXTURIZER_HPP

#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/geometry/Normal.hpp"

#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace lvr2
{

/**
 * @brief A texturizer that uses images instead of pointcloud colors for creating the textures
 *        for meshes.
 */
template<typename BaseVecT>
class ImageTexturizer : public Texturizer<BaseVecT> 
{

public:

    /**
     * @brief constructor
     */
    ImageTexturizer(
        float texelSize,
        int minClusterSize,
        int maxClusterSize
    ) : Texturizer<BaseVecT>(texelSize, minClusterSize, maxClusterSize)
    {
        image_data_initialized = false;
    }

    /**
     * @brief Sets the internal UOS Scanproject to the given one
     *
     * @param project The UOS Scanproject the intern project will be set to.
     */
    void set_project(ScanProject& project)
    {
        this->project = project;
    }

    /**
     * @brief Generates a Texture for a given Rectangle
     *
     * @param index The newly created texture will get this index.
     *
     * @param surface Unused in this Texturizer
     *
     * @param boudingRect The texture will be generated for this rectangle
     *
     * @return Returns a handle for the newly created texture.
     */
    virtual TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
    ) override;

private:
    /// @cond internal
    ScanProject project;

    bool image_data_initialized;
    std::vector<ScanImage> images;

    void init_image_data();

    template<typename ValueType>
    void undistorted_to_distorted_uv(ValueType &u, ValueType &v, const ScanImage &img);

    bool exclude_image(BaseVecT pos, const ScanImage &image_data);

    bool point_behind_camera(BaseVecT pos, const ScanImage &image_data);
    /// @endcond
};

} // namespace lvr2

#include "ImageTexturizer.tcc"

#endif

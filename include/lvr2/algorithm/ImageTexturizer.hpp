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
 * ImageTexturizer.hpp
 *
 *  @date 01.11.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_ALGORITHM_IMAGETEXTURIZER_HPP
#define LVR2_ALGORITHM_IMAGETEXTURIZER_HPP

#include <lvr2/algorithm/Texturizer.hpp>
#include <lvr2/geometry/Normal.hpp>

#include <lvr2/io/ScanprojectIO.hpp>
#include <lvr2/geometry/Matrix4.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace lvr2
{

/// @cond internal
template<typename BaseVecT>
struct ImageData {
    cv::Mat data;
    Vector<BaseVecT>  pos;
    Vector<BaseVecT>  dir;
    Matrix4<BaseVecT> project_to_image_transform;
    float distortion_params[6];
    float intrinsic_params[4];
};
/// @endcond

/**
 * @brief A texturizer that uses images instead of pointcloud colors for creating the textures
 *        for meshes.
 */
template<typename BaseVecT>
class ImageTexturizer : public Texturizer<BaseVecT> {

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
    void set_project(Scanproject& project)
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
    TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<BaseVecT>& boundingRect
    );

private:
    /// @cond internal
    Scanproject project;

    bool image_data_initialized;
    std::vector<ImageData<BaseVecT> > images;

    void init_image_data();

    template<typename ValueType>
    void undistorted_to_distorted_uv(ValueType &u, ValueType &v, const ImageData<BaseVecT> &img);

    bool exclude_image(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data);

    bool point_behind_camera(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data);
    /// @endcond
};

} // namespace lvr2

#include "ImageTexturizer.tcc"

#endif

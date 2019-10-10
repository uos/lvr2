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
 * ColorAlgorithms.hpp
 *
 * Collection of algorithms for color calculation.
 *
 * @date 21.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_COLORALGORITHMS_H_
#define LVR2_ALGORITHM_COLORALGORITHMS_H_

#include <vector>
#include <array>

#include <boost/optional.hpp>



#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"

namespace lvr2
{

using Rgb8Color = std::array<uint8_t, 3>;

/**
 * @brief   Calculates the color of each vertex from the point cloud
 *
 * For each vertex, its color is calculated from the rgb color information in
 * the meshes surface.
 *
 * @param   mesh    The mesh
 * @param   surface The surface of the mesh
 *
 * @return  Optional of a DenseVertexMap with a Rgb8Color for each vertex
 */
template<typename BaseVecT>
boost::optional<DenseVertexMap<Rgb8Color>> calcColorFromPointCloud(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurfacePtr<BaseVecT> surface
);

/**
 * @brief   Convert a given float to an 8-bit RGB-Color, using the rainbowcolor scale.
 *
 * The given float, which we want to convert to an 8-bit RGB-Color, has to be in [0, 1].
 * If it is bigger than 1 it's value is set to 1.
 * If it is smaller than 0 it's value is set to 0.
 *
 * @param   value The float value, which will be converted to an RGB-Rainbowcolor.
 *
 * @return  The 8-bit RGB-Color, interpreted as rainbowcolor.
 */
static Rgb8Color floatToRainbowColor(float value);

/**
 * @brief   Convert a given float to an 8-bit Grayscale-Color.
 *
 * The given float, which we want to convert to an 8-bit GrayScale-Color, has to be in [0, 1].
 * If it is bigger than 1 it's value is set to 1.
 * If it is smaller than 0 it's value is set to 0.
 *
 * @param   value The float value, which will be converted to a Grayscale-Color.
 *
 * @return  The 8-bit Grayscale-Color.
 */
static Rgb8Color floatToGrayScaleColor(float value);

/**
 * @brief    Calculate the color for the centroid of a given face
 *
 *           For a given mesh and it's surface the color of the faces centroid
 *           is calculated. The face is identified by the given face handle.
 *
 * @param    mesh     The mesh
 * @param    surface  The surface of the mesh
 * @param    faceH    Face handle of the face
 *
 * @return   The Rgb8Color of the centroid
 */
template<typename BaseVecT>
Rgb8Color calcColorForFaceCentroid(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurface<BaseVecT>& surface,
    FaceHandle faceH
);

} // namespace lvr2

#include "lvr2/algorithm/ColorAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_COLORALGORITHMS_H_ */

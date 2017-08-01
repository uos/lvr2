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
#include <boost/optional.hpp>

using std::vector;
using boost::optional;

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>

namespace lvr2
{

using Rgb8Color = array<uint8_t, 3>;

template<typename BaseVecT>
optional<DenseVertexMap<Rgb8Color>> calcColorFromPointCloud(
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
Rgb8Color floatToRainbowColor(float value);

/**
 * @brief   Convert a given float to an 8-bit Grayscale-Color.
 *
 * @param   value The float value, which will be converted to a Grayscale-Color.
 *
 * @return  The 8-bit Grayscale-Color.
 */
Rgb8Color floatToGrayScaleColor(float value);

} // namespace lvr2

#include <lvr2/algorithm/ColorAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_COLORALGORITHMS_H_ */

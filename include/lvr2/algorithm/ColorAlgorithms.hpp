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
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

using Rgb8Color = array<uint8_t, 3>;

template<typename BaseVecT>
optional<VertexMap<Rgb8Color>> calcColorFromPointCloud(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurfacePtr<BaseVecT> surface
);

} // namespace lvr2

#include <lvr2/algorithm/ColorAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_COLORALGORITHMS_H_ */

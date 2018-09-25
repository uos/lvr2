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

/**
 * EigenSVDPointAlign.hpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#ifndef EIGENSVDPOINTALIGN_HPP_
#define EIGENSVDPOINTALIGN_HPP_

#include <lvr2/io/PointBuffer2.hpp>
#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/Vector.hpp>

namespace lvr2
{

template <typename BaseVecT>
using PointPairVector = std::vector<std::pair<Vector<BaseVecT>, Vector<BaseVecT>> >;

template <typename BaseVecT>
class EigenSVDPointAlign
{
public:
    EigenSVDPointAlign() {};
    double alignPoints(
            const PointPairVector<BaseVecT>& pairs,
            const Vector<BaseVecT> centroid1,
            const Vector<BaseVecT> centroid2,
            Matrix4<BaseVecT>& align);
};

} /* namespace lvr2 */

#include "EigenSVDPointAlign.cpp"

#endif /* EIGENSVDPOINTALIGN_HPP_ */

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
 * ClusterPainter.hpp
 *
 *  @date 18.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_CLUSTERPAINTER_H_
#define LVR2_ALGORITHM_CLUSTERPAINTER_H_

#include <cstdint>
#include <array>

using std::array;

#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/ClusterSet.hpp>
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

/**
 * @brief Algorithm which generates the same color for all vertices, which are in the same cluster.
 */
class ClusterPainter
{
public:
    using Rgb8Color = array<uint8_t, 3>;
    ClusterPainter(const ClusterSet<FaceHandle>& clusterSet) : m_clusterSet(clusterSet) {};

    template<typename BaseVecT>
    VertexMap<Rgb8Color> simpsons(const BaseMesh<BaseVecT>& mesh) const;
private:
    ClusterSet<FaceHandle> m_clusterSet;
    Rgb8Color getSimpsonColorForIdx(size_t idx) const;
};

} // namespace lvr2

#include <lvr2/algorithm/ClusterPainter.tcc>

#endif /* LVR2_ALGORITHM_CLUSTERPAINTER_H_ */

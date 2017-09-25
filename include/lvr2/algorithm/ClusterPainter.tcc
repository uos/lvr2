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
 * ClusterPainter.tcc
 *
 *  @date 18.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <cmath>

using std::cos;
using std::sin;
using std::fabs;

namespace lvr2
{

template<typename BaseVecT>
DenseClusterMap<Rgb8Color> ClusterPainter::simpsons(const BaseMesh<BaseVecT>& mesh) const
{
    DenseClusterMap<Rgb8Color> colorMap;
    colorMap.reserve(m_clusterBiMap.numHandles() * 3);
    size_t clusterIdx = 0;
    for (auto clusterH: m_clusterBiMap)
    {
        auto color = getSimpsonColorForIdx(clusterIdx);
        colorMap.insert(clusterH, color);

        clusterIdx++;
    }

    return colorMap;
}

} // namespace lvr2

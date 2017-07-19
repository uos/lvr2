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

#include <cmath>

using std::cos;
using std::sin;
using std::fabs;

namespace lvr2
{

template<typename BaseVecT>
VertexMap<ClusterPainter::Rgb8Color> ClusterPainter::simpsons(const BaseMesh<BaseVecT>& mesh) const
{
    VertexMap<Rgb8Color> colorMap;
    size_t clusterIdx = 0;
    for (auto clusterH: m_clusterSet)
    {
        auto color = getSimpsonColorForIdx(clusterIdx);

        for (auto faceH: m_clusterSet[clusterH].handles)
        {
            auto vertices = mesh.getVertexHandlesOfFace(faceH);

            colorMap.insert(vertices[0], color);
            colorMap.insert(vertices[1], color);
            colorMap.insert(vertices[2], color);
        }

        clusterIdx++;
    }

    return colorMap;
}

template<typename BaseVecT>
VertexMap<ClusterPainter::Rgb8Color> ClusterPainter::fromPointCloud(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurfacePtr<BaseVecT> surface) const
{

    int k = 1; // k-nearest-neighbors

    VertexMap<Rgb8Color> vertexMap;

    for (auto vertexH: mesh.vertices())
    {
        vector<size_t> cv;
        Point<BaseVecT> p = mesh.getVertexPosition(vertexH);
        surface->searchTree().kSearch(p, k, cv);

        float r = 0.0f, g = 0.0f, b = 0.0f;

        for (size_t pointIdx : cv)
        {
            // TODO: optional?? was passiert wenn keine farbe da ist?

            array<uint8_t,3> colors = *(surface->pointBuffer()->getRgbColor(pointIdx));
            r += colors[0];
            g += colors[1];
            b += colors[2];
        }

        r /= k;
        g /= k;
        b /= k;

        vertexMap.insert(vertexH, {
                static_cast<uint8_t>(r),
                static_cast<uint8_t>(g),
                static_cast<uint8_t>(b)
            });
    }

    return vertexMap;
}

ClusterPainter::Rgb8Color ClusterPainter::getSimpsonColorForIdx(size_t idx) const
{
    return {
        static_cast<uint8_t>(fabs(cos(idx)) * 255),
        static_cast<uint8_t>(fabs(sin(idx * 30)) * 255),
        static_cast<uint8_t>(fabs(sin(idx * 2)) * 255)
    };
}

} // namespace lvr2

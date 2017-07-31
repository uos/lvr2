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
 * ColorAlgorithms.tcc
 *
 * @date 21.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <array>

using std::array;

#include <lvr2/geometry/Point.hpp>

namespace lvr2
{

template <typename BaseVecT>
optional<DenseVertexMap<Rgb8Color>> calcColorFromPointCloud(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurfacePtr<BaseVecT> surface
)
{
    if (!surface->pointBuffer()->hasRgbColor())
    {
        return boost::none;
    }

    DenseVertexMap<Rgb8Color> vertexMap;
    vertexMap.reserve(mesh.numVertices());

    // k-nearest-neighbors
    const int k = 1;

    vector<size_t> cv;
    for (auto vertexH: mesh.vertices())
    {
        cv.clear();
        auto p = mesh.getVertexPosition(vertexH);
        surface->searchTree().kSearch(p, k, cv);

        float r = 0.0f, g = 0.0f, b = 0.0f;

        for (size_t pointIdx : cv)
        {
            auto colors = *(surface->pointBuffer()->getRgbColor(pointIdx));
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

Rgb8Color floatToRainbowColor(float value)
{
    Rgb8Color result_color;
    value = std::min(value, 1.0f);
    value = std::max(value, 0.0f);

    float h = value * 5.0f + 1.0f;
    int i = floor(h);
    float f = h - i;
    if ( !(i&1) ) f = 1 - f; // if i is even
    float n = 1 - f;

    if (i <= 1)
    {
        result_color[0] = floor(n * 255);
        result_color[1] = 0;
        result_color[2] = 255;
    }
    else if (i == 2)
    {
        result_color[0] = 0;
        result_color[1] = floor(n * 255);
        result_color[2] = 255;
    }
    else if (i == 3)
    {
        result_color[0] = 0;
        result_color[1] = 255;
        result_color[2] = floor(n * 255);
    }
    else if (i == 4)
    {
        result_color[0] = floor(n * 255);
        result_color[1] = 255;
        result_color[2] = 0;
    }
    else if (i >= 5)
    {
        result_color[0] = 255;
        result_color[1] = floor(n * 255);
        result_color[2] = 0;
    }

    return result_color;
}

Rgb8Color floatToGrayScaleColor(float value)
{
    std::array<uint8_t, 3> return_color = {0, 0, 0};

    //if (max == min) return return_color; //avoid to divide by 0

    int grayscale_result = 255 * (value);
    return_color[0] = grayscale_result;
    return_color[1] = grayscale_result;
    return_color[2] = grayscale_result;

    return return_color;
}



} // namespace lvr2

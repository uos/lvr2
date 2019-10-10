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
 * ColorAlgorithms.tcc
 *
 * @date 21.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <algorithm>
#include <cmath>

using std::array;

namespace lvr2
{

template <typename BaseVecT>
boost::optional<DenseVertexMap<Rgb8Color>> calcColorFromPointCloud(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurfacePtr<BaseVecT> surface
)
{
    if (!surface->pointBuffer()->hasColors())
    {
        // cout << "none" << endl;
        return boost::none;
    }

    DenseVertexMap<Rgb8Color> vertexMap;
    vertexMap.reserve(mesh.numVertices());

    // k-nearest-neighbors
    const int k = 1;

    UCharChannel colors = *(surface->pointBuffer()->getUCharChannel("colors"));

    vector<size_t> cv;
    for (auto vertexH: mesh.vertices())
    {
        cv.clear();
        auto p = mesh.getVertexPosition(vertexH);
        surface->searchTree()->kSearch(p, k, cv);

        float r = 0.0f, g = 0.0f, b = 0.0f;

        for (size_t pointIdx : cv)
        {
            auto color = colors[pointIdx];
            r += color[0];
            g += color[1];
            b += color[2];
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

static Rgb8Color floatToRainbowColor(float value)
{
    value = std::min(value, 1.0f);
    value = std::max(value, 0.0f);

    //TODO: understand and fix style
    float h = value * 5.0f + 1.0f;
    int i = floor(h);
    float f = h - i;

    // if i is even
    if (i % 2 == 0)
    {
        f = 1 - f;
    }

    float n = 1 - f;

    if (i <= 1)
    {
        return { static_cast<uint8_t>(floor(n * 255)), 0, 255 };
    }
    else if (i == 2)
    {
        return { 0, static_cast<uint8_t>(floor(n * 255)), 255 };
    }
    else if (i == 3)
    {
        return { 0, 255, static_cast<uint8_t>(floor(n * 255)) };
    }
    else if (i == 4)
    {
        return { static_cast<uint8_t>(floor(n * 255)), 255, 0 };
    }
    else if (i >= 5)
    {
        return { 255, static_cast<uint8_t>(floor(n * 255)), 0 };
    }
    else 
    {
    return { 255, 255, 255 };
    }
}

static Rgb8Color floatToGrayScaleColor(float value)
{
    if(value > 1)
    {
        value = 1;
    }
    if(value < 0)
    {
        value = 0;
    }
    int grayscaleResult = 255 * (value);

    return {
        static_cast<uint8_t>(grayscaleResult),
        static_cast<uint8_t>(grayscaleResult),
        static_cast<uint8_t>(grayscaleResult)
    };
}


template<typename BaseVecT>
Rgb8Color calcColorForFaceCentroid(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurface<BaseVecT>& surface,
    FaceHandle faceH
)
{
    if (surface.pointBuffer()->hasColors())
    {
        vector<size_t> cv;
        auto centroid = mesh.calcFaceCentroid(faceH);
        UCharChannel colors = *(surface.pointBuffer()->getUCharChannel("colors"));

        // Find color of face centroid
        int k = 1; // k-nearest-neighbors
        surface.searchTree()->kSearch(centroid, k, cv);
        uint8_t r = 0, g = 0, b = 0;
        for (size_t pointIdx : cv)
        {
            auto cur_color = colors[pointIdx];
            r += cur_color[0];
            g += cur_color[1];
            b += cur_color[2];
        }
        r /= k;
        g /= k;
        b /= k;

        // "Smooth" colors: convert 0:255 to 0:1, round to 2 decimal places, convert back
        // For better re-using of a single color later on
        Rgb8Color color = {
            static_cast<uint8_t>((floor((((float)r)/255.0)*100.0+0.5)/100.0) * 255.0),
            static_cast<uint8_t>((floor((((float)g)/255.0)*100.0+0.5)/100.0) * 255.0),
            static_cast<uint8_t>((floor((((float)b)/255.0)*100.0+0.5)/100.0) * 255.0)
        };

        return color;
    }
    else
    {
        Rgb8Color color = {0,0,0};
        return color;
    }
}

} // namespace lvr2

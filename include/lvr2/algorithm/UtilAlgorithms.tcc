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

#include <algorithm>

#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/geometry/Handles.hpp"

namespace lvr2
{

template<
    template<typename, typename> class OutMapT,
    typename InMapT,
    typename MapF
>
OutMapT<typename InMapT::HandleType, std::result_of_t<MapF(typename InMapT::ValueType)>> map(
    const InMapT& mapIn,
    MapF func
)
{
    // FIXME: oh god, the C++ template verbosity. I mean, this function works
    // and is pretty generic. You could improve code size a bit, but I will
    // leave it as is ;-)
    static_assert(
        std::is_base_of<
            AttributeMap<typename InMapT::HandleType, typename InMapT::ValueType>,
            InMapT
        >::value,
        "The `mapIn` argument needs to be an AttributeMap!"
    );

    static_assert(
        std::is_base_of<
            AttributeMap<
                typename InMapT::HandleType,
                std::result_of_t<MapF(typename InMapT::ValueType)>
            >,
            OutMapT<
                typename InMapT::HandleType,
                std::result_of_t<MapF(typename InMapT::ValueType)>
            >
        >::value,
        "The `OutMapT` template argument needs to be an AttributeMap!"
    );

    OutMapT<
        typename InMapT::HandleType,
        std::result_of_t<MapF(typename InMapT::ValueType)>
    > resultMap;
    resultMap.reserve(mapIn.numValues());

    for (auto vH: mapIn)
    {
        resultMap.insert(vH, func(mapIn[vH]));
    }

    return resultMap;
}

template<
    template<typename, typename> class OutMapT,
    typename IterProxyT,
    typename GenF
>
OutMapT<
    typename decltype(std::declval<IterProxyT>().begin())::HandleType,
    typename std::result_of<GenF(typename decltype(std::declval<IterProxyT>().begin())::HandleType)>::type
>
    attrMapFromFunc(
        IterProxyT iterProxy,
        GenF func
)
{
    OutMapT<
        typename decltype(std::declval<IterProxyT>().begin())::HandleType,
        typename std::result_of<GenF(typename decltype(std::declval<IterProxyT>().begin())::HandleType)>::type
    > out;

    for (auto handle: iterProxy)
    {
        out.insert(handle, func(handle));
    }
    return out;
}

template<typename HandleT, typename ValueT>
pair<ValueT, ValueT> minMaxOfMap(const AttributeMap<HandleT, ValueT>& map)
{
    ValueT min;
    ValueT max;

    for (auto handle: map)
    {
        min = std::min(min, map[handle]);
        max = std::max(max, map[handle]);
    }

    return make_pair(min, max);
}



void rasterize_line(Eigen::Vector2f p0, Eigen::Vector2f p1, std::function<void(Eigen::Vector2i)> cb)
{
    using Eigen::Vector2i;
    float x_dist = std::abs( p1.x() - p0.x());
    float y_dist = std::abs( p1.y() - p0.y());
    // Plot along x
    if (x_dist > y_dist)
    {
        Eigen::Vector2f left, right;
        left = p0.x() < p1.x() ? p0 : p1;
        right = p0.x() < p1.x() ? p1 : p0;

        // Check for div by 0
        if (x_dist == 0.0f) 
        {
            cb(left.cast<int>());
            return;
        }

        float slope = ((float) right.y() - left.y()) / ((float) right.x() - left.x());
        // Left point
        cb(left.cast<int>());
        // y at the right edge of the pixel containing the left point (use floor(x + 1) in case x is a whole number)
        float y_0 = left.y() + slope * (std::floor(left.x() + 1.0f) - left.x());
        // Check if y_0 is between floor(left.y()) and ceil(left.y())
        if (((int) left.y()) != ((int) y_0))
        {
            cb(Vector2i(left.x(), y_0));
        }

        // Right point
        cb(right.cast<int>());
        // y at the left edge of the pixel containing the right point
        float y_1 = right.y() - slope * (right.x() - std::floor(right.x()));
        // Check if y_1 is between floor(right.y()) and ceil(right.y())
        if (((int) right.y()) != ((int) y_1))
        {
            cb(Vector2i(right.x(), y_1));
        }

        // The line
        float y = y_0;
        for (int x = floor(left.x() + 1.0f); x < floor(right.x()); x++)
        {
            float next_y = y + slope;
            cb(Vector2i(x, y));

            if (((int) y) != ((int) next_y))
            {
                cb(Vector2i(x, next_y));
            }

            y = next_y;
        }
  
    }
    // Plot along y
    else
    {
        Eigen::Vector2f down, up;
        down = p0.y() < p1.y() ? p0 : p1;
        up = p0.y() < p1.y() ? p1 : p0;

        // Check for div by 0
        if (y_dist == 0.0f) 
        {
            cb(down.cast<int>());
            return;
        }

        float slope = ((float) up.x() - down.x()) / ((float) up.y() - down.y());

        // low point
        cb(down.cast<int>());
        // x at the upper edge of the pixel containing the lower point
        float x_0 = down.x() + slope * (std::floor(down.y() + 1) - down.y());
        // Check if x_0 is between floor(down.x()) and ceil(down.x())
        if (((int) down.x()) != ((int) x_0))
        {
            cb(Vector2i(x_0, down.y()));
        }

        // high point
        cb(up.cast<int>());
        // x at the lower edge of the pixel containing the high point
        float x_1 = up.x() - slope * (up.y() - std::floor(up.y()));
        // Check if x_1 is between floor(up.x()) and ceil(up.x())
        if (((int) up.x()) != ((int) x_1))
        {
            cb(Vector2i(x_1, up.y()));
        }
        
        // The line
        float x = x_0;
        for (int y = floor(down.y() + 1.0f); y < floor(up.y()); y++)
        {
            float next_x = x + slope;
            cb(Vector2i(x, y));

            if (((int) x) != ((int) next_x))
            {
                cb(Vector2i(next_x, y));
            }

            x = next_x;
        }
    }

}

} // namespace lvr2

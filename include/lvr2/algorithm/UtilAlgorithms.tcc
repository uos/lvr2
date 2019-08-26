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

} // namespace lvr2

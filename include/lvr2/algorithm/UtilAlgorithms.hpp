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

//
// Created by aaltemoeller on 01.08.17.
//

#ifndef LAS_VEGAS_UTILALGORITHMS_HPP
#define LAS_VEGAS_UTILALGORITHMS_HPP

#include <iterator>
#include <type_traits>
#include <utility>

#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/geometry/Handles.hpp"


namespace lvr2 {

/**
 * @brief Calls `func` for each value of the given map and save the result in
 *        the output map.
 *
 * The type of the output map needs to be specified explicitly as template
 * parameter. You typically call it like this:
 *
 * \code{.cpp}
 * auto vertexColors = map<DenseAttrMap>(vertexCosts, [](float vertexCost)
 * {
 *     // Convert float vertex-cost to color...
 * })
 * \endcode
 *
 * @tparam MapT The (rank-2) type of the attribute map implementation used
 *              for the output map. E.g. `DenseAttrMap` or `SparseAttrMap`.
 */
template<
    template<typename, typename> typename OutMapT,
    typename InMapT,
    typename MapF
>
OutMapT<typename InMapT::HandleType, std::result_of_t<MapF(typename InMapT::ValueType)>> map(
    const InMapT& mapIn,
    MapF func
);

/**
 * @brief Creates an attribute map by calling the given function for each
 *        element in the given iterator. The iterator yields the keys, the
 *        function gives us the corresponding value.
 */
template<
    template<typename, typename> typename OutMapT,
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
);

/**
 * @brief Returns the minimum and maximum element from the given map.
 *
 * Of course, this assumes that the values in the map are comparable with the
 * standard comparison operators.
 */
template<typename HandleT, typename ValueT>
pair<ValueT, ValueT> minMaxOfMap(const AttributeMap<HandleT, ValueT>& map);

} // namespace lvr2

#include "lvr2/algorithm/UtilAlgorithms.tcc"

#endif //LAS_VEGAS_UTILALGORITHMS_HPP

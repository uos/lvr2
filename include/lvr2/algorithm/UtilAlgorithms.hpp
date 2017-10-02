//
// Created by aaltemoeller on 01.08.17.
//

#ifndef LAS_VEGAS_UTILALGORITHMS_HPP
#define LAS_VEGAS_UTILALGORITHMS_HPP

#include <iterator>
#include <type_traits>
#include <utility>

#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>


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

#include <lvr2/algorithm/UtilAlgorithms.tcc>

#endif //LAS_VEGAS_UTILALGORITHMS_HPP

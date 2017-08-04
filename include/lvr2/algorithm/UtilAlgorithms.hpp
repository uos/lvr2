//
// Created by aaltemoeller on 01.08.17.
//

#ifndef LAS_VEGAS_UTILALGORITHMS_HPP
#define LAS_VEGAS_UTILALGORITHMS_HPP

#include <type_traits>

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
 * @tparam MapT The (class-2) type of the attribute map implementation used
 *               for the output map. E.g. `DenseAttrMap` or `SparseAttrMap`.
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

} // namespace lvr2

#include <lvr2/algorithm/UtilAlgorithms.tcc>

#endif //LAS_VEGAS_UTILALGORITHMS_HPP

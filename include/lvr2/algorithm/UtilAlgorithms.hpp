//
// Created by aaltemoeller on 01.08.17.
//

#ifndef LAS_VEGAS_UTILALGORITHMS_HPP
#define LAS_VEGAS_UTILALGORITHMS_HPP

#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>


namespace lvr2 {

/**
 * @brief   Change the given input VertexMap to a different type output DenseVertexMap.
 *
 * The conversion between the input VertexMap and output DenseVertexMap is defined by the given function.
 * The method changeMap takes every element of the input VertexMap and converts it individually via the given
 * map-conversion function and adds the converted element to the output DenseVertexMap.
 *
 * @tparam  in              Templatetype for the input VertexMap.
 * @tparam  out             Templatetype for the output DenseVertexMap.
 * @tparam  MapF            Templatetype for the given map-conversion function.
 * @param   map_in          VertexMap holding input-type data, which will be
 *                          converted to the DenseVertexMap holding output-type data.
 * @param   map_function    Function for converting a single element of the
 *                          input-type to a single element of the output-type.
 *
 * @return  A DenseVertexMap holding output-type data,
 *          which is created via the input VertexMap and the map-conversion function.
 */
template<typename in, typename out, typename MapF>
DenseVertexMap<out> map(const VertexMap <in> &map_in, MapF map_function);

} // namespace lvr2

#include <lvr2/algorithm/UtilAlgorithms.tcc>

#endif //LAS_VEGAS_UTILALGORITHMS_HPP

#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

template<typename in, typename out, typename MapF>
DenseVertexMap <out> map(const VertexMap <in> &map_in, MapF map_function) {
    DenseVertexMap <out> resultMap;

    for (auto vH: map_in)
    {
        resultMap.insert(vH, map_function(map_in[vH]));
    }

    return resultMap;
}

} // namespace lvr2

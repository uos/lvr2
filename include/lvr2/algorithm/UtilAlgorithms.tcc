#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

template<typename in, typename out, typename MapF>
DenseVertexMap <out> map(const VertexMap<in> &mapIn, MapF mapFunction) {
    DenseVertexMap <out> resultMap;

    for (auto vH: mapIn)
    {
        resultMap.insert(vH, mapFunction(mapIn[vH]));
    }

    return resultMap;
}

} // namespace lvr2

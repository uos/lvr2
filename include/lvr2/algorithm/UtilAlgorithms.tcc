#include <algorithm>

#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

template<
    template<typename, typename> typename OutMapT,
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

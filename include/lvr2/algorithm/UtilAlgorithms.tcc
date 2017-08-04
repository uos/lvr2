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

} // namespace lvr2

#ifndef LVR2_UTIL_TUPLE_HPP
#define LVR2_UTIL_TUPLE_HPP

#include <type_traits>
#include <tuple>

namespace lvr2 {

template <typename T1, typename... T2>
constexpr bool tupleContains(std::tuple<T2...>) {
    return std::disjunction_v<std::is_same<T1, T2>...>;
}

template<typename... Tp>
struct Tuple : std::tuple<Tp...> 
{
    template<typename T>
    static constexpr bool contains()
    {
        return std::disjunction_v<std::is_same<T, Tp>...>;
    }
};

} // namespace lvr2

#endif // LVR2_UTIL_TUPLE_HPP
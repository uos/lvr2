#ifndef LVR2_UTIL_TYPE_TRAITS_HPP
#define LVR2_UTIL_TYPE_TRAITS_HPP

#include <type_traits>

namespace lvr2 {

/**
 * @brief This type trait was necessary to 
 * 
 * Usage:
 * 
 * @code
 * auto v = vector[0];
 * 
 * static_assert(
 *      arg_has_type<Vector<float> >(v),
 *      "Error: Type mismatch"
 * );
 * 
 * @endcode
 * 
 * Why?
 * The following code snipped was compiling in .cpp files but not in .cu
 * - Ubuntu 20, GCC 9.4.0, NVCC 12.4
 * 
 * @code
 * auto v = vector[0];
 * 
 * static_assert(
 *      std::is_same<decltype(v), BaseVector<float> >::value,
 *      "Error: Type mismatch"
 * );
 * @endcode
 * 
 * Since cuda code sometimes includes BaseVector.hpp all cuda libraries wont compile.
 * Problem was that `decltype` in .cu file is not working as expected
 * 
*/
template<typename T, typename AutoType>
static constexpr bool arg_has_type(const AutoType& t1)
{
    return std::is_same<T, AutoType>::value;
}

} // namespace lvr2

#endif // LVR2_UTIL_TYPE_TRAITS_HPP
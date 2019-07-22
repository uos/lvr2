#pragma once

#include <unordered_map>
#include <iostream>
#include "VariantChannel.hpp"

namespace lvr2 {

template<typename... T>
class VariantChannelMap
: public std::unordered_map<std::string, VariantChannel<T...> > 
{

protected:
    template <class T1, class Tuple>
    struct TupleIndex;

public:
    using key_type = std::string;
    using val_type = VariantChannel<T...>;
    using types = std::tuple<T...>;

    /**
     * @brief Access type index with type
     * - example: ChanneVariantMap<int, float> my_map;
     *            ChanneVariantMap<int, float>::type_index<int>::value -> 0
     */
    template<class U>
    struct index_of_type {
        static const std::size_t value = TupleIndex<U, types>::value;
    };

    template <std::size_t N>
    using type_of_index = typename std::tuple_element<N, types>::type;

    /**
     * @brief Add an Key + AttributeChannel to the map
     * 
     */
    template<typename U>
    void add(const std::string& name, Channel<U> channel);

    /**
     * @brief Get AttributeChannel with type U from map as reference
     * 
     */
    template<typename U>
    Channel<U>& get(const std::string& name);

    /**
     * @brief get AttributeChannel with type U from map
     * 
     */
    template<typename U>
    const Channel<U>& get(const std::string& name) const;

    /**
     * @brief get type index of a map entry
     * 
     */
    int type(const std::string& name) const;

    /**
     * @brief Check if key has specific type U
     *          Example: cm.is_type<float>("points") -> true
     */
    template<typename U>
    bool is_type(const std::string& name) const;

    /**
     * @brief Output cout
     * 
     */
    friend std::ostream& operator<<(std::ostream& os, const VariantChannelMap<T...>& cm)
    {
        std::cout << "[VariantChannelMap]\n";
        for(auto it : cm)
        {
            std::cout << it.first << ": " << it.second  << "\n";
        }
        return os;
    }

protected:
    template <class T1, class... Types>
    struct TupleIndex<T1, std::tuple<T1, Types...>> {
        static const std::size_t value = 0;
    };

    template <class T1, class U, class... Types>
    struct TupleIndex<T1, std::tuple<U, Types...>> {
        static const std::size_t value = 1 + TupleIndex<T1, std::tuple<Types...>>::value;
    };

};

} // namespace lvr2

#include "VariantChannelMap.tcc"


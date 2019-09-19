#pragma once

#ifndef LVR2_TYPES_VARIANTCHANNEL
#define LVR2_TYPES_VARIANTCHANNEL

#include <boost/variant.hpp>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "Channel.hpp"

namespace lvr2 {


template<typename... T>
class VariantChannel : public boost::variant<Channel<T>...>
{
    using base = boost::variant<Channel<T>...>;
    using base::base;
protected:
    template <class T1, class Tuple>
    struct TupleIndex;

public:
    using types = std::tuple<T...>;

    /**
     * @brief Access type index with type
     * - example: ChanneVariantMap<int, float> my_map;
     *            ChanneVariantMap<int, float>::type_index<int>::value -> 0
     */
    template<class U>
    struct index_of_type {
        static constexpr std::size_t value = TupleIndex<U, types>::value;
    };

    static constexpr std::size_t num_types = std::tuple_size<types>::value;

    template <std::size_t N>
    using type_of_index = typename std::tuple_element<N, types>::type;

    size_t numElements() const;

    size_t width() const;

    template<typename U>
    boost::shared_array<U> dataPtr() const;

    template<std::size_t N>
    boost::shared_array<type_of_index<N> > dataPtr() const
    {
        return boost::apply_visitor(DataPtrVisitor<type_of_index<N> >(), *this);
    }

    /**
     * @brief Get type index of a map entry
     * 
     */
    int type() const;

    template<typename U>
    Channel<U> extract() const;

    template<typename U>
    Channel<U>& extract();

    /**
     * @brief Checks if key has specific type U.
     * @example cm.is_type<float>() -> true
     */
    template<typename U>
    bool is_type() const;

    friend std::ostream& operator<<(std::ostream& os, const VariantChannel<T...>& ch)
    {
        os << "type: " << ch.type() << ", " << static_cast <const base &>(ch);
        return os;
    }

    VariantChannel<T...> clone() const;
    
// Visitor Implementations
protected:
    struct NumElementsVisitor : public boost::static_visitor<size_t>
    {
        template<typename U>
        size_t operator()(const Channel<U>& channel) const
        {
            return channel.numElements();
        }
    };

    struct WidthVisitor : public boost::static_visitor<size_t>
    {
        template<typename U>
        size_t operator()(const Channel<U>& channel) const
        {
            return channel.width();
        }
    };

    template<typename U>
    struct DataPtrVisitor : public boost::static_visitor<boost::shared_array<U> >
    {
        template<typename V,
                std::enable_if_t<std::is_same<U, V>::value, int> = 0>
        boost::shared_array<U> operator()(const Channel<V>& channel) const
        {
            return channel.dataPtr();
        }

        template<typename V,
                std::enable_if_t<!std::is_same<U, V>::value, int> = 0>
        boost::shared_array<U> operator()(const Channel<V>& channel) const
        {
            throw std::invalid_argument("tried to get wrong type of channel");
            return boost::shared_array<U>();
        }
    };

    struct CloneVisitor : public boost::static_visitor< VariantChannel<T...> >
    {
        template<typename U>
        VariantChannel<T...> operator()(const Channel<U>& channel) const
        {
            return channel.clone();
        }
    };

    template <class T1, class... Types>
    struct TupleIndex<T1, std::tuple<T1, Types...>> {
        static constexpr std::size_t value = 0;
    };

    template <class T1, class U, class... Types>
    struct TupleIndex<T1, std::tuple<U, Types...>> {
        static constexpr std::size_t value = 1 + TupleIndex<T1, std::tuple<Types...>>::value;
    };

};

template<typename ...Tp>
using VariantChannelOptional = boost::optional<VariantChannel<Tp...> >;


} // namespace lvr2

#include "VariantChannel.tcc"

#endif // LVR2_TYPES_VARIANTCHANNEL

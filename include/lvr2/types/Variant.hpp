/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LVR2_TYPES_VARIANT_HPP
#define LVR2_TYPES_VARIANT_HPP

#include <boost/variant.hpp>
#include <boost/type_index.hpp>
#include <tuple>
#include <iostream>
#include <memory>

namespace lvr2
{

/**
 * @brief LVR2 variant type. Based on boost::variant with some extra functions
 * 
 * @tparam T 
 */
template<typename... T>
class Variant : public boost::variant<T...>
{
    using base = boost::variant<T...>;
    using base::base;

protected:
    template <class T1, class Tuple>
    struct TupleIndex;

public:
    using types = std::tuple<T...>;
    using base::which;

    template<typename U>
    static constexpr std::size_t index_of_type()
    {
        return TupleIndex<U, types>::value;
    }

    static constexpr std::size_t num_types() 
    {
        return std::tuple_size<types>::value;
    }
    // static constexpr std::size_t num_types = std::tuple_size<types>::value;

    template <std::size_t N>
    using type_of_index = typename std::tuple_element<N, types>::type;

    std::string typeName() const;

    int type() const;

    template<typename U>
    U get() const;

    template<typename U>
    U& get();

    /**
     * @brief Checks if key has specific type U.
     * @example cm.is_type<float>() -> true
     */
    template<typename U>
    constexpr bool is_type() const
    {
        return this->which() == index_of_type<U>();
    }

    friend std::ostream& operator<<(std::ostream& os, const Variant<T...>& v)
    {
        // os << "type: " << ch.typeName() << ", " << static_cast <const base &>(ch);
        return os;
    }

// Visitor Implementations
protected:
    struct TypeNameVisitor : public boost::static_visitor<std::string>
    {
        template<typename U>
        std::string operator()(const U& type) const
        {
            return boost::typeindex::type_id<U>().pretty_name();
        }
    };

    // Helper
    template <class T1, class... Types>
    struct TupleIndex<T1, std::tuple<T1, Types...>> {
        static constexpr std::size_t value = 0;
    };

    template <class T1, class U, class... Types>
    struct TupleIndex<T1, std::tuple<U, Types...>> {
        static constexpr std::size_t value = 1 + TupleIndex<T1, std::tuple<Types...>>::value;
    };
};

template<typename T, typename... Tp>
T& operator<<=(T& x, const Variant<Tp...>& v)
{
    x = v.template get<T>();
    return x;
}

} // namespace lvr2

#include "Variant.tcc"

#endif // LVR2_TYPES_VARIANT_HPP
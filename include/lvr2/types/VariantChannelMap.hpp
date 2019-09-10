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

#pragma once

#ifndef LVR2_TYPES_VARIANTCHANNELMAP
#define LVR2_TYPES_VARIANTCHANNELMAP

#include <unordered_map>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include "VariantChannel.hpp"

namespace lvr2 {

template<typename... T>
class VariantChannelMap
: public std::unordered_map<std::string, VariantChannel<T...> > 
{
public:
    using key_type = std::string;
    using val_type = VariantChannel<T...>;
    using elem_type = std::pair<const key_type, val_type>;

    using types = std::tuple<T...>;
    using base = std::unordered_map<std::string, VariantChannel<T...> >;
    using base::base;

    /**
     * @brief Access type index by type.
     * @details Example usage: ChanneVariantMap<int, float> my_map;
     *            ChanneVariantMap<int, float>::type_index<int>::value -> 0
     */
    template<class U>
    struct index_of_type {
        static constexpr std::size_t value = val_type::template index_of_type<U>::value;
    };

    template <std::size_t N>
    using type_of_index = typename val_type::template type_of_index<N>;

    static constexpr std::size_t num_types = val_type::num_types;

    template<typename U>
    struct iterator {

        using resolved_elem_type = std::pair<const key_type&, Channel<U>& >;
        using pointer = std::shared_ptr<resolved_elem_type>;
        // using pointer = elem_type*;
        using reference = elem_type&;

        iterator(
            typename base::iterator base_it,
            typename base::iterator end_it)
        :m_base_it(base_it),
        m_end_it(end_it)
        {
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
        }

        resolved_elem_type operator*() const noexcept
        {
            return {
                m_base_it->first,
                m_base_it->second.template extract<U>()
            };
        }

        pointer operator->() const noexcept
        {
            return pointer(
                new resolved_elem_type({
                    m_base_it->first,
                    m_base_it->second.template extract<U>()
                })
            );
        }

        iterator<U>& operator++() noexcept
        {
            m_base_it++;
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
            return *this;
        }

        iterator<U> operator++(int) noexcept
        {
            iterator<U> tmp(*this);
            m_base_it++;
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
            return tmp;
        }

        inline bool operator==(const typename base::iterator& rhs) noexcept
        {
            return m_base_it == rhs;
        }

        inline bool operator!=(const typename base::iterator& rhs) noexcept
        {
            return m_base_it != rhs;
        }

        typename base::iterator operator()() const noexcept
        {
            return m_base_it;
        }

        typename base::iterator m_base_it;
        typename base::iterator m_end_it;
    };


    template<typename U>
    struct const_iterator {

        using resolved_elem_type = std::pair<const key_type&, const Channel<U>& >;
        using pointer = std::shared_ptr<resolved_elem_type>;
        // using pointer = elem_type*;
        using reference = elem_type&;

        const_iterator(
            typename base::const_iterator base_it,
            typename base::const_iterator end_it)
        :m_base_it(base_it),
        m_end_it(end_it)
        {
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
        }

        resolved_elem_type operator*() const noexcept
        {
            return {
                m_base_it->first,
                m_base_it->second.template extract<U>()
            };
        }

        pointer operator->() const noexcept
        {
            return pointer(
                new resolved_elem_type({
                    m_base_it->first,
                    m_base_it->second.template extract<U>()
                })
            );
        }

        const_iterator<U>& operator++() noexcept
        {
            m_base_it++;
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
            return *this;
        }

        const_iterator<U> operator++(int) noexcept
        {
            iterator<U> tmp(*this);
            m_base_it++;
            while(m_base_it != m_end_it && m_base_it->second.which() != index_of_type<U>::value)
            {
                m_base_it++;
            }
            return tmp;
        }

        inline bool operator==(const typename base::const_iterator& rhs) noexcept
        {
            return m_base_it == rhs;
        }

        inline bool operator!=(const typename base::const_iterator& rhs) noexcept
        {
            return m_base_it != rhs;
        }

        typename base::iterator operator()() const noexcept
        {
            return m_base_it;
        }

        typename base::const_iterator m_base_it;
        typename base::const_iterator m_end_it;
    };

    /**
     * @brief Adds an Key + AttributeChannel to the map.
     * 
     * @param[in] name Key of the channel.
     * @param[in] channel The channel.
     * 
     */
    template<typename U>
    void add(const std::string& name, Channel<U> channel);


    /**
     * @brief Adds an empty channel
     * @param[in] name Key of the channel.
     * 
     */
    template<typename U>
    void add(const std::string& name);

    /**
     * @brief Adds an empty channel with size
     * @param[in] name Key of the channel.
     * @param[in] numElements Number of elements in channel.
     * @param[in] width Element size.
     * 
     */
    template<typename U>
    void add(const std::string& name, size_t numElements, size_t width);

    /**
     * @brief Gets AttributeChannel with type U from map as reference.
     * 
     * @param[in] name Key of the channel.
     */
    template<typename U>
    Channel<U>& get(const std::string& name);

    /**
     * @brief Gets AttributeChannel by type U from map.
     * 
     * @param[in] name Key of the channel.
     * 
     */
    template<typename U>
    const Channel<U>& get(const std::string& name) const;


    template<typename U>
    typename Channel<U>::Optional getOptional(const std::string& name);

    template<typename U>
    const typename Channel<U>::Optional getOptional(const std::string& name) const;

    /**
     * @brief Gets type index of a map entry.
     * 
     * @param[in] Key of the channel.
     * @return Index of type tuple of the variant.
     */
    int type(const std::string& name) const;

    /**
     * @brief Checks if key has specific type U.
     * @details Example Usage: cm.is_type<float>("points") -> true
     * 
     * @param[in] name Key of the channel.
     * @tparam U Type of the value to check.
     * @return true If the type is equal.
     * @return false If the type is unequal.
     */
    template<typename U>
    bool is_type(const std::string& name) const;

    /**
     * @brief Gets the available keys by a specific type.
     * 
     * @tparam U Type of the channels.
     * @return Vector of keys.
     * 
     */
    template<typename U>
    std::vector<std::string> keys();

    /**
     * @brief Counts the number of channels by a specific type.
     * @detail For total number of channels use "size()"
     * 
     * @tparam U Type of the channels.
     * @return Number of channels.
     * 
     */
    template<typename U>
    size_t numChannels();

    template<typename U>
    iterator<U> typedBegin()
    {
        typename base::iterator it_base = this->begin();
        typename base::iterator it_end = this->end();
        return iterator<U>(it_base, it_end);
    }

    template<typename U>
    const_iterator<U> typedBegin() const
    {
        typename base::const_iterator it_base = this->begin();
        typename base::const_iterator it_end = this->end();
        return const_iterator<U>(it_base, it_end);
    }

    using base::erase;

    template<typename U>
    iterator<U> erase(iterator<U> it)
    {
        typename base::iterator it_base = erase(it());
        typename base::iterator it_end = this->end();
        return iterator<U>(it_base, it_end);
    }

    template<typename V>
    VariantChannelMap<T...> manipulate(V visitor)
    {
        VariantChannelMap<T...> cm;
        for(auto vchannel: *this)
        {
            cm.insert({vchannel.first, boost::apply_visitor(visitor, vchannel.second)});
        }
        return cm;
    }

    VariantChannelMap<T...> clone() const;

    /**
     * @brief Output cout
     * 
     */
    friend std::ostream& operator<<(std::ostream& os, const VariantChannelMap<T...>& cm)
    {
        std::cout << "[ VariantChannelMap ]\n";
        for(auto it : cm)
        {
            std::cout << it.first << ": " << it.second  << "\n";
        }
        return os;
    }
};

} // namespace lvr2

#include "VariantChannelMap.tcc"

#endif // LVR2_TYPES_VARIANTCHANNELMAP


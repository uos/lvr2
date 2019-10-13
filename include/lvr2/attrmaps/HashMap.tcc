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

/*
 * HashMap.tcc
 *
 *  @date 27.07.2017
 */
#include <utility>
#include <memory>
#include "lvr2/util/Panic.hpp"

using std::make_pair;

namespace lvr2
{

template<typename HandleT, typename ValueT>
HashMap<HandleT, ValueT>::HashMap(const ValueT& defaultValue)
    : m_default(defaultValue)
{}

template<typename HandleT, typename ValueT>
HashMap<HandleT, ValueT>::HashMap(size_t countElements, const ValueT& defaultValue)
    : m_default(defaultValue)
{
    reserve(countElements);
}

template<typename HandleT, typename ValueT>
bool HashMap<HandleT, ValueT>::containsKey(HandleT key) const
{
    return m_map.find(key) != m_map.end();
}

template<typename HandleT, typename ValueT>
boost::optional<ValueT> HashMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
{
    auto res = m_map.insert(make_pair(key, value));
    if (!res.second)
    {
        // TODO: this makes some copies that are not necessary. Dunno how
        // to correctly code this right now. Maybe the compiler optimizes
        // everything perfectly anyway.
        auto old = (*res.first).second;
        (*res.first).second = value;
        return old;
    }
    else
    {
        return boost::none;
    }
}

template<typename HandleT, typename ValueT>
boost::optional<ValueT> HashMap<HandleT, ValueT>::erase(HandleT key)
{
    auto it = m_map.find(key);
    if (it != m_map.end())
    {
        auto out = (*it).second;
        m_map.erase(it);
        return out;
    }
    else
    {
        return boost::none;
    }
}

template<typename HandleT, typename ValueT>
void HashMap<HandleT, ValueT>::clear()
{
    m_map.clear();
}

template<typename HandleT, typename ValueT>
boost::optional<ValueT&> HashMap<HandleT, ValueT>::get(HandleT key)
{
    // Try to lookup value. If none was found and a default value is set,
    // insert it and return that instead.
    auto it = m_map.find(key);
    if (it == m_map.end())
    {
        if (m_default)
        {
            // Insert default value into hash map and return the inserted value
            auto res = m_map.insert(make_pair(key, *m_default));
            return (*res.first).second;
        }
        else
        {
            return boost::none;
        }
    }
    return (*it).second;
}

template<typename HandleT, typename ValueT>
boost::optional<const ValueT&> HashMap<HandleT, ValueT>::get(HandleT key) const
{
    // Try to lookup value. If none was found and a default value is set,
    // return that instead.
    auto it = m_map.find(key);
    if (it == m_map.end())
    {
        if (m_default)
        {
            return *m_default;
        }
        else
        {
            return boost::none;
        }
    }
    return (*it).second;
}

template<typename HandleT, typename ValueT>
size_t HashMap<HandleT, ValueT>::numValues() const
{
    return m_map.size();
}


template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> HashMap<HandleT, ValueT>::begin() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<HashMapIterator<HandleT, ValueT>>(m_map.begin())
    );
}

template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> HashMap<HandleT, ValueT>::end() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<HashMapIterator<HandleT, ValueT>>(m_map.end())
    );
}

template<typename HandleT, typename ValueT>
void HashMap<HandleT, ValueT>::reserve(size_t newCap)
{
    m_map.reserve(newCap);
}

template<typename HandleT, typename ValueT>
HashMapIterator<HandleT, ValueT>::HashMapIterator(
    typename unordered_map<HandleT, ValueT>::const_iterator iter
)
    : m_iter(iter)
{}

template<typename HandleT, typename ValueT>
AttributeMapHandleIterator<HandleT>& HashMapIterator<HandleT, ValueT>::operator++()
{
    ++m_iter;
    return *this;
}

template<typename HandleT, typename ValueT>
bool HashMapIterator<HandleT, ValueT>::operator==(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const HashMapIterator<HandleT, ValueT>*>(&other);
    return cast && m_iter == cast->m_iter;
}

template<typename HandleT, typename ValueT>
bool HashMapIterator<HandleT, ValueT>::operator!=(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const HashMapIterator<HandleT, ValueT>*>(&other);
    return !cast || m_iter != cast->m_iter;
}

template<typename HandleT, typename ValueT>
HandleT HashMapIterator<HandleT, ValueT>::operator*() const
{
    return (*m_iter).first;
}

template<typename HandleT, typename ValueT>
std::unique_ptr<AttributeMapHandleIterator<HandleT>> HashMapIterator<HandleT, ValueT>::clone() const
{
    return std::make_unique<HashMapIterator>(*this);
}



} // namespace lvr2

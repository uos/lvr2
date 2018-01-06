/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * HashMap.tcc
 *
 *  @date 27.07.2017
 */
#include <utility>

#include <lvr2/util/Panic.hpp>

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
optional<ValueT> HashMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
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
optional<ValueT> HashMap<HandleT, ValueT>::erase(HandleT key)
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
optional<ValueT&> HashMap<HandleT, ValueT>::get(HandleT key)
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
optional<const ValueT&> HashMap<HandleT, ValueT>::get(HandleT key) const
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

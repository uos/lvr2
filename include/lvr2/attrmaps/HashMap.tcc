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
bool HashMap<HandleT, ValueT>::containsKey(HandleT key) const
{
    return m_map.find(key) != m_map.end();
}

template<typename HandleT, typename ValueT>
optional<ValueT> HashMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
{
    auto out = remove(key);
    m_map.insert(make_pair(key, value));
    return out;
}

template<typename HandleT, typename ValueT>
optional<ValueT> HashMap<HandleT, ValueT>::remove(HandleT key)
{
    auto elem = get(key);
    if (elem)
    {
        auto out = *elem;
        m_map.erase(key);
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
    auto it = m_map.find(key);
    if (it == m_map.end())
    {
        return boost::none;
    }
    return (*it).second;
}

template<typename HandleT, typename ValueT>
optional<const ValueT&> HashMap<HandleT, ValueT>::get(HandleT key) const
{
    auto it = m_map.find(key);
    if (it == m_map.end())
    {
        return boost::none;
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
HashMapIterator<HandleT, ValueT>::HashMapIterator(typename unordered_map<HandleT, ValueT>::const_iterator iter)
    : m_iter(iter)
{}

template<typename HandleT, typename ValueT>
AttributeMapHandleIterator<HandleT>& HashMapIterator<HandleT, ValueT>::operator++()
{
    ++m_iter;
    return *this;
}

template<typename HandleT, typename ValueT>
bool HashMapIterator<HandleT, ValueT>::operator==(const AttributeMapHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HashMapIterator<HandleT, ValueT>*>(&other);
    return cast && m_iter == cast->m_iter;
}

template<typename HandleT, typename ValueT>
bool HashMapIterator<HandleT, ValueT>::operator!=(const AttributeMapHandleIterator<HandleT>& other) const
{
    auto cast = dynamic_cast<const HashMapIterator<HandleT, ValueT>*>(&other);
    return cast && m_iter != cast->m_iter;
}

template<typename HandleT, typename ValueT>
HandleT HashMapIterator<HandleT, ValueT>::operator*() const
{
    return (*m_iter).first;
}



} // namespace lvr2

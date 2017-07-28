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
 * ListMap.tcc
 *
 *  @date 27.07.2017
 */

#include <algorithm>

#include <lvr2/util/Panic.hpp>


namespace lvr2
{

template<typename HandleT, typename ValueT>
ListMap<HandleT, ValueT>::ListMap(const ValueT& defaultValue)
    : m_default(defaultValue)
{}

template<typename HandleT, typename ValueT>
ListMap<HandleT, ValueT>::ListMap(size_t countElements, const ValueT& defaultValue)
    : m_default(defaultValue)
{
    reserve(countElements);
}

template<typename HandleT, typename ValueT>
typename vector<pair<HandleT, ValueT>>::const_iterator
    ListMap<HandleT, ValueT>::keyIterator(HandleT key) const
{
    return std::find_if(m_list.begin(), m_list.end(), [&](auto& elem)
    {
        return elem.first == key;
    });
}

template<typename HandleT, typename ValueT>
typename vector<pair<HandleT, ValueT>>::iterator ListMap<HandleT, ValueT>::keyIterator(HandleT key)
{
    return std::find_if(m_list.begin(), m_list.end(), [&](auto& elem)
    {
        return elem.first == key;
    });
}

template<typename HandleT, typename ValueT>
bool ListMap<HandleT, ValueT>::containsKey(HandleT key) const
{
    return keyIterator(key) != m_list.end();
}

template<typename HandleT, typename ValueT>
optional<ValueT> ListMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
{
    if (numValues() > 256)
    {
        // panic(
        //     "More than 256 items in a tiny map! This implementation is not "
        //         "designed to handle anything but a tiny number of values. If "
        //         "you think this panic is too pedantic, just remove it..."
        // );
    }
    auto out = remove(key);
    m_list.push_back(make_pair(key, value));
    return out;
}

template<typename HandleT, typename ValueT>
optional<ValueT> ListMap<HandleT, ValueT>::remove(HandleT key)
{
    auto it = keyIterator(key);
    if (it == m_list.end())
    {
        return boost::none;
    }
    else
    {
        auto out = (*it).second;
        m_list.erase(it);
        return out;
    }
}

template<typename HandleT, typename ValueT>
void ListMap<HandleT, ValueT>::clear()
{
    m_list.clear();
}

template<typename HandleT, typename ValueT>
optional<ValueT&> ListMap<HandleT, ValueT>::get(HandleT key)
{
    // Try to lookup value. If none was found and a default value is set,
    // insert it and return that instead.
    auto it = keyIterator(key);
    if (it == m_list.end())
    {
        if (m_default)
        {
            // Insert default value into hash map and return the inserted value
            m_list.push_back(make_pair(key, *m_default));
            return m_list.back().second;
        }
        else
        {
            return boost::none;
        }
    }
    return (*it).second;
}

template<typename HandleT, typename ValueT>
optional<const ValueT&> ListMap<HandleT, ValueT>::get(HandleT key) const
{
    // Try to lookup value. If none was found and a default value is set,
    // return that instead.
    auto it = keyIterator(key);
    if (it == m_list.end())
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
size_t ListMap<HandleT, ValueT>::numValues() const
{
    return m_list.size();
}


template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> ListMap<HandleT, ValueT>::begin() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<ListMapIterator<HandleT, ValueT>>(m_list.begin())
    );
}

template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> ListMap<HandleT, ValueT>::end() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<ListMapIterator<HandleT, ValueT>>(m_list.end())
    );
}

template<typename HandleT, typename ValueT>
void ListMap<HandleT, ValueT>::reserve(size_t newCap)
{
    m_list.reserve(newCap);
}

template<typename HandleT, typename ValueT>
ListMapIterator<HandleT, ValueT>::ListMapIterator(
    typename vector<pair<HandleT, ValueT>>::const_iterator iter
)
    : m_iter(iter)
{}

template<typename HandleT, typename ValueT>
AttributeMapHandleIterator<HandleT>& ListMapIterator<HandleT, ValueT>::operator++()
{
    ++m_iter;
    return *this;
}

template<typename HandleT, typename ValueT>
bool ListMapIterator<HandleT, ValueT>::operator==(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const ListMapIterator<HandleT, ValueT>*>(&other);
    return cast && m_iter == cast->m_iter;
}

template<typename HandleT, typename ValueT>
bool ListMapIterator<HandleT, ValueT>::operator!=(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const ListMapIterator<HandleT, ValueT>*>(&other);
    return !cast || m_iter != cast->m_iter;
}

template<typename HandleT, typename ValueT>
HandleT ListMapIterator<HandleT, ValueT>::operator*() const
{
    return (*m_iter).first;
}


} // namespace lvr2

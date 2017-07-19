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
 * VectorMap.tcc
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <lvr2/util/Panic.hpp>

namespace lvr2
{

template<typename KeyT, typename ValT>
VectorMap<KeyT, ValT>::VectorMap(size_t countElements, const ValueType& defaultValue)
    : m_vec(countElements, Wrapper(defaultValue))
{}

template<typename KeyT, typename ValT>
void VectorMap<KeyT, ValT>::insert(const KeyType& key, const ValueType& value)
{
    // Check if elements vector is large enough
    // TODO: enhance insert -> delete stuff
    while (m_vec.size() <= key.idx())
    {
        auto h = m_vec.push_back(Wrapper());
        m_vec.erase(h);
    }

    m_vec.set(key, value);
}

template<typename KeyT, typename ValT>
typename VectorMap<KeyT, ValT>::Wrapper& VectorMap<KeyT, ValT>::Wrapper::operator=(const Wrapper& value)
{
    new(&data) ValueType(value.data);
    return *this;
}

template<typename KeyT, typename ValT>
void VectorMap<KeyT, ValT>::erase(const KeyType& key)
{
    m_vec.erase(key);
}

template<typename KeyT, typename ValT>
boost::optional<const ValT&> VectorMap<KeyT, ValT>::get(const KeyType& key) const
{
    auto maybe = m_vec.get(key);
    if (maybe)
    {
        return maybe->data;
    }
    return boost::none;
}

template<typename KeyT, typename ValT>
boost::optional<ValT&> VectorMap<KeyT, ValT>::get(const KeyType& key)
{
    auto maybe = m_vec.get(key);
    if (maybe)
    {
        return maybe->data;
    }
    return boost::none;
}

template<typename KeyT, typename ValT>
ValT& VectorMap<KeyT, ValT>::operator[](const KeyType& key)
{
    return m_vec[key].data;
}

template<typename KeyT, typename ValT>
const ValT& VectorMap<KeyT, ValT>::operator[](const KeyType& key) const
{
    return m_vec[key].data;
}

template<typename KeyT, typename ValT>
size_t VectorMap<KeyT, ValT>::sizeUsed() const
{
    return m_vec.sizeUsed();
}

template<typename KeyT, typename ValT>
decltype(auto) VectorMap<KeyT, ValT>::begin() const
{
    return m_vec.begin();
}
template<typename KeyT, typename ValT>
decltype(auto) VectorMap<KeyT, ValT>::end() const
{
    return m_vec.end();
}


} // namespace lvr2

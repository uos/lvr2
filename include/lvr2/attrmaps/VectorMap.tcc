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

#include <utility>

using std::move;

#include <lvr2/util/Panic.hpp>

namespace lvr2
{

template<typename HandleT, typename ValueT>
VectorMap<HandleT, ValueT>::VectorMap(const ValueT& defaultValue)
    : m_default(defaultValue)
{}

template<typename HandleT, typename ValueT>
VectorMap<HandleT, ValueT>::VectorMap(size_t countElements, const ValueT& defaultValue)
    : m_default(defaultValue)
{
    reserve(countElements);
}

template<typename HandleT, typename ValueT>
bool VectorMap<HandleT, ValueT>::containsKey(HandleT key) const
{
    return static_cast<bool>(m_vec.get(key));
}

template<typename HandleT, typename ValueT>
optional<ValueT> VectorMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
{
    // If the vector isn't large enough yet, we allocate additional space.
    if (key.idx() >= m_vec.size())
    {
        m_vec.increaseSize(key);
        m_vec.push(value);
        return boost::none;
    }
    else
    {
        auto out = erase(key);
        m_vec.set(key, value);
        return out;
    }
}

template<typename HandleT, typename ValueT>
optional<ValueT> VectorMap<HandleT, ValueT>::erase(HandleT key)
{
    auto val = m_vec.get(key);
    if (val)
    {
        auto out = ValueT(move(*val));
        m_vec.erase(key);
        return out;
    }
    else
    {
        return boost::none;
    }
}

template<typename HandleT, typename ValueT>
void VectorMap<HandleT, ValueT>::clear()
{
    m_vec.clear();
}

template<typename HandleT, typename ValueT>
optional<ValueT&> VectorMap<HandleT, ValueT>::get(HandleT key)
{
    // Try to lookup value. If none was found and a default value is set,
    // insert it and return that instead.
    auto res = m_vec.get(key);
    if (!m_vec.get(key) && m_default)
    {
        insert(key, *m_default);
        return m_vec.get(key);
    }
    return res;
}

template<typename HandleT, typename ValueT>
optional<const ValueT&> VectorMap<HandleT, ValueT>::get(HandleT key) const
{
    // Try to lookup value. If none was found and a default value is set,
    // return that instead.
    auto res = m_vec.get(key);
    return (!m_vec.get(key) && m_default) ? *m_default : res;
}

template<typename HandleT, typename ValueT>
size_t VectorMap<HandleT, ValueT>::numValues() const
{
    return m_vec.numUsed();
}

template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> VectorMap<HandleT, ValueT>::begin() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<VectorMapIterator<HandleT, ValueT>>(m_vec.begin())
    );
}

template<typename HandleT, typename ValueT>
AttributeMapHandleIteratorPtr<HandleT> VectorMap<HandleT, ValueT>::end() const
{
    return AttributeMapHandleIteratorPtr<HandleT>(
        std::make_unique<VectorMapIterator<HandleT, ValueT>>(m_vec.end())
    );
}

template<typename HandleT, typename ValueT>
void VectorMap<HandleT, ValueT>::reserve(size_t newCap)
{
    m_vec.reserve(newCap);
};


template<typename HandleT, typename ValueT>
VectorMapIterator<HandleT, ValueT>::VectorMapIterator(StableVectorIterator<HandleT, ValueT> iter)
    : m_iter(iter)
{}

template<typename HandleT, typename ValueT>
AttributeMapHandleIterator<HandleT>& VectorMapIterator<HandleT, ValueT>::operator++()
{
    ++m_iter;
    return *this;
}

template<typename HandleT, typename ValueT>
bool VectorMapIterator<HandleT, ValueT>::operator==(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const VectorMapIterator<HandleT, ValueT>*>(&other);
    return cast && m_iter == cast->m_iter;
}

template<typename HandleT, typename ValueT>
bool VectorMapIterator<HandleT, ValueT>::operator!=(
    const AttributeMapHandleIterator<HandleT>& other
) const
{
    auto cast = dynamic_cast<const VectorMapIterator<HandleT, ValueT>*>(&other);
    return !cast || m_iter != cast->m_iter;
}

template<typename HandleT, typename ValueT>
HandleT VectorMapIterator<HandleT, ValueT>::operator*() const
{
    return *m_iter;
}

} // namespace lvr2
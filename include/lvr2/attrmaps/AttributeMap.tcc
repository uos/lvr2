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
 * AttributeMap.tcc
 *
 *  @date 26.07.2017
 */

#include <lvr2/util/Panic.hpp>

namespace lvr2
{

template<typename HandleT, typename ValueT>
ValueT& AttributeMap<HandleT, ValueT>::operator[](HandleT key)
{
    auto elem = get(key);
    if (!elem)
    {
        panic("attempt to access a non-existing value in an attribute map");
    }
    return *elem;
}

template<typename HandleT, typename ValueT>
const ValueT& AttributeMap<HandleT, ValueT>::operator[](HandleT key) const
{
    auto elem = get(key);
    if (!elem)
    {
        panic("attempt to access a non-existing value in an attribute map");
    }
    return *elem;
}

template<typename HandleT>
AttributeMapHandleIteratorPtr<HandleT>& AttributeMapHandleIteratorPtr<HandleT>::operator++()
{
    ++(*m_iter);
    return *this;
}

template<typename HandleT>
bool AttributeMapHandleIteratorPtr<HandleT>::operator==(const AttributeMapHandleIteratorPtr& other) const
{
    return *m_iter == *other.m_iter;
}

template<typename HandleT>
bool AttributeMapHandleIteratorPtr<HandleT>::operator!=(const AttributeMapHandleIteratorPtr& other) const
{
    return *m_iter != *other.m_iter;
}

template<typename HandleT>
HandleT AttributeMapHandleIteratorPtr<HandleT>::operator*() const
{
    return **m_iter;
}


} // namespace lvr2

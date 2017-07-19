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
 * StableVector.tcc
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <lvr2/util/Panic.hpp>

namespace lvr2
{

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::checkAccess(const HandleType& handle) const
{
    // You cannot access deleted or uninitialized elements!
    if (m_deleted[handle.idx()])
    {
        panic("attempt to access a deleted value in StableVector");
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(size_t countElements, const ElemT& defaultValue)
    : m_elements(countElements, defaultValue),
      m_deleted(countElements, false),
      m_usedCount(countElements)
{}


template<typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::push_back(const ElementType& elem)
{
    m_elements.push_back(elem);
    m_deleted.push_back(false);
    ++m_usedCount;
    return HandleT(size() - 1);
}

template <typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::nextHandle() const
{
    return HandleT(size());
}


template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::erase(const HandleType& handle)
{
    checkAccess(handle);

    m_deleted[handle.idx()] = true;
    --m_usedCount;
}

template<typename HandleT, typename ElemT>
boost::optional<ElemT&> StableVector<HandleT, ElemT>::get(const HandleType& key)
{
    if (m_deleted[key.idx()])
    {
        return boost::none;
    }
    return m_elements[key.idx()];
}

template<typename HandleT, typename ElemT>
boost::optional<const ElemT&> StableVector<HandleT, ElemT>::get(const HandleType& key) const
{
    if (m_deleted[key.idx()])
    {
        return boost::none;
    }
    return m_elements[key.idx()];
}

template<typename HandleT, typename ElemT>
ElemT& StableVector<HandleT, ElemT>::operator[](const HandleType& handle)
{
    checkAccess(handle);

    return m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
const ElemT& StableVector<HandleT, ElemT>::operator[](const HandleType& handle) const
{
    checkAccess(handle);

    return m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
size_t StableVector<HandleT, ElemT>::size() const
{
    return m_deleted.size();
}

template<typename HandleT, typename ElemT>
size_t StableVector<HandleT, ElemT>::sizeUsed() const
{
    return m_usedCount;
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::set(const HandleType& key, const ElementType& elem)
{
    // check access
    if (key.idx() >= m_elements.size())
    {
        panic("attempt to append new element in StableVector with set -> use push_back!");
    }

    // insert element
    if (m_deleted[key.idx()])
    {
        ++m_usedCount;
    }
    m_elements[key.idx()] = elem;
    m_deleted[key.idx()] = false;
};

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::reserve(size_t newCap)
{
    m_elements.reserve(newCap);
    m_deleted.reserve(newCap);
};

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT> StableVector<HandleT, ElemT>::begin() const
{
    return StableVectorIterator<HandleT>(&this->m_deleted);
}

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT> StableVector<HandleT, ElemT>::end() const
{
    return StableVectorIterator<HandleT>(&this->m_deleted, true);
}

template <typename HandleT>
StableVectorIterator<HandleT>::StableVectorIterator(const vector<bool>* deleted, bool startAtEnd)
    : m_deleted(deleted), m_pos(startAtEnd ? deleted->size() : 0)
{
    if (m_pos == 0 && !m_deleted->empty() && (*m_deleted)[0])
    {
        ++(*this);
    }
}

template<typename HandleT>
StableVectorIterator<HandleT>& StableVectorIterator<HandleT>::operator=(const StableVectorIterator<HandleT>& other)
{
    if (&other == this)
    {
        return *this;
    }
    m_pos = other.m_pos;
    m_deleted = other.m_deleted;

    return *this;
}

template<typename HandleT>
bool StableVectorIterator<HandleT>::operator==(const StableVectorIterator<HandleT>& other) const
{
    return m_pos == other.m_pos && m_deleted == other.m_deleted;
}

template<typename HandleT>
bool StableVectorIterator<HandleT>::operator!=(const StableVectorIterator<HandleT>& other) const
{
    return !(*this == other);
}

template<typename HandleT>
StableVectorIterator<HandleT>& StableVectorIterator<HandleT>::operator++()
{
    // If not at the end, advance by one element
    if (m_pos < m_deleted->size())
    {
        m_pos++;
    }

    // Advance until the next element, at least 1 element behind the vector, to
    // indicate the end of iteration.
    while (m_pos < m_deleted->size() && (*m_deleted)[m_pos])
    {
        m_pos++;
    }

    return *this;
}

template<typename HandleT>
HandleT StableVectorIterator<HandleT>::operator*() const
{
    return HandleT(m_pos);
}

} // namespace lvr2

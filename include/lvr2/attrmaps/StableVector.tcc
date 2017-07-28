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

#include <sstream>
#include <string>


namespace lvr2
{

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::checkAccess(HandleType handle) const
{
    // Make sure the handle is not OOB
    if (handle.idx() >= size())
    {
        std::stringstream ss;
        ss << "lookup with an out of bounds handle (" << handle.idx() << ") in StableVector";
        panic(ss.str());
    }

    // You cannot access deleted or uninitialized elements!
    if (!m_elements[handle.idx()])
    {
        panic("attempt to access a deleted value in StableVector");
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(size_t countElements, const ElementType& defaultValue)
    : m_elements(countElements, defaultValue),
      m_usedCount(countElements)
{}

template<typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::push(const ElementType& elem)
{
    m_elements.emplace_back(elem);
    ++m_usedCount;
    return HandleT(size() - 1);
}

template<typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::push(ElementType&& elem)
{
    m_elements.emplace_back(move(elem));
    ++m_usedCount;
    return HandleT(size() - 1);
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::increaseSize(HandleType upTo)
{
    if (upTo.idx() < size())
    {
        panic("call to increaseSize() with a valid handle!");
    }

    m_elements.resize(upTo.idx(), boost::none);
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::increaseSize(HandleType upTo, const ElementType& elem)
{
    if (upTo.idx() < size())
    {
        panic("call to increaseSize() with a valid handle!");
    }

    m_elements.resize(upTo.idx(), elem);
}

template <typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::nextHandle() const
{
    return HandleT(size());
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::erase(HandleType handle)
{
    checkAccess(handle);

    m_elements[handle.idx()] = boost::none;
    --m_usedCount;
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::clear()
{
    m_elements.clear();
}

template<typename HandleT, typename ElemT>
boost::optional<ElemT&> StableVector<HandleT, ElemT>::get(HandleType handle)
{
    if (handle.idx() >= size() || !m_elements[handle.idx()])
    {
        return boost::none;
    }
    return *m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
boost::optional<const ElemT&> StableVector<HandleT, ElemT>::get(HandleType handle) const
{
    if (handle.idx() >= size() || !m_elements[handle.idx()])
    {
        return boost::none;
    }
    return *m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
ElemT& StableVector<HandleT, ElemT>::operator[](HandleType handle)
{
    checkAccess(handle);
    return *m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
const ElemT& StableVector<HandleT, ElemT>::operator[](HandleType handle) const
{
    checkAccess(handle);
    return *m_elements[handle.idx()];
}

template<typename HandleT, typename ElemT>
size_t StableVector<HandleT, ElemT>::size() const
{
    return m_elements.size();
}

template<typename HandleT, typename ElemT>
size_t StableVector<HandleT, ElemT>::numUsed() const
{
    return m_usedCount;
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::set(HandleType handle, const ElementType& elem)
{
    // check access
    if (handle.idx() >= size())
    {
        panic("attempt to append new element in StableVector with set() -> use push()!");
    }

    // insert element
    if (!m_elements[handle.idx()])
    {
        ++m_usedCount;
    }
    m_elements[handle.idx()] = elem;
};

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::set(HandleType handle, ElementType&& elem)
{
    // check access
    if (handle.idx() >= size())
    {
        panic("attempt to append new element in StableVector with set() -> use push()!");
    }

    // insert element
    if (!m_elements[handle.idx()])
    {
        ++m_usedCount;
    }
    m_elements[handle.idx()] = elem;
};

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::reserve(size_t newCap)
{
    m_elements.reserve(newCap);
};

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT, ElemT> StableVector<HandleT, ElemT>::begin() const
{
    return StableVectorIterator<HandleT, ElemT>(&this->m_elements);
}

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT, ElemT> StableVector<HandleT, ElemT>::end() const
{
    return StableVectorIterator<HandleT, ElemT>(&this->m_elements, true);
}

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT, ElemT>::StableVectorIterator(
    const vector<optional<ElemT>>* deleted,
    bool startAtEnd
)
    : m_elements(deleted), m_pos(startAtEnd ? deleted->size() : 0)
{
    if (m_pos == 0 && !m_elements->empty() && !(*m_elements)[0])
    {
        ++(*this);
    }
}

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT, ElemT>& StableVectorIterator<HandleT, ElemT>::operator=(
    const StableVectorIterator<HandleT, ElemT>& other
)
{
    if (&other == this)
    {
        return *this;
    }
    m_pos = other.m_pos;
    m_elements = other.m_elements;

    return *this;
}

template<typename HandleT, typename ElemT>
bool StableVectorIterator<HandleT, ElemT>::operator==(
    const StableVectorIterator<HandleT, ElemT>& other
) const
{
    return m_pos == other.m_pos && m_elements == other.m_elements;
}

template<typename HandleT, typename ElemT>
bool StableVectorIterator<HandleT, ElemT>::operator!=(
    const StableVectorIterator<HandleT, ElemT>& other
) const
{
    return !(*this == other);
}

template<typename HandleT, typename ElemT>
StableVectorIterator<HandleT, ElemT>& StableVectorIterator<HandleT, ElemT>::operator++()
{
    // If not at the end, advance by one element
    if (m_pos < m_elements->size())
    {
        m_pos++;
    }

    // Advance until the next element, at most 1 element behind the vector, to
    // indicate the end of iteration.
    while (m_pos < m_elements->size() && !(*m_elements)[m_pos])
    {
        m_pos++;
    }

    return *this;
}

template<typename HandleT, typename ElemT>
HandleT StableVectorIterator<HandleT, ElemT>::operator*() const
{
    return HandleT(m_pos);
}

} // namespace lvr2

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
    if (m_deleted[handle.idx()])
    {
        panic("attempt to access a deleted value in StableVector");
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::~StableVector()
{
    // Manually call the destructor of all values that are not deleted.
    for (auto handle: *this)
    {
        m_elements[handle.idx()].data.~ElementType();
    }

    // When this method ends, both vectors are destroyed. This, however, will
    // only free their memory, since neither `bool` nor `UnsafeWrapper` have
    // a destructor that does anything.
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(size_t countElements, const ElementType& defaultValue)
    : m_elements(countElements, UnsafeWrapper()),
      m_deleted(countElements, false),
      m_usedCount(countElements)
{
    for (auto& elem: m_elements)
    {
        elem.set(defaultValue);
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(const StableVector& other)
    : m_elements(other.size(), UnsafeWrapper()),
      m_deleted(other.m_deleted),
      m_usedCount(other.m_usedCount)
{
    // Right now, we correctly created the `m_deleted` and `m_usedCount`
    // fields. The `m_elements` field was already allocated and filled with
    // default constructed `UnsafeWrapper` elements (random garbage). We now
    // need to copy over all non-deleted values.
    for (auto handle: other)
    {
        m_elements[handle.idx()].set(other[handle]);
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(StableVector&& other)
    : m_elements(other.size(), UnsafeWrapper()),
      m_deleted(other.m_deleted),
      m_usedCount(other.m_usedCount)
{
    // Right now, we correctly created the `m_deleted` and `m_usedCount`
    // fields. The `m_elements` field was already allocated and filled with
    // default constructed `UnsafeWrapper` elements (random garbage). We now
    // need to copy over all non-deleted values.
    for (auto handle: other)
    {
        m_elements[handle.idx()].set(std::move(other[handle]));
    }
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>& StableVector<HandleT, ElemT>::operator=(const StableVector& other)
{
    this->~StableVector();
    new (this) StableVector(other);
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>& StableVector<HandleT, ElemT>::operator=(StableVector&& other)
{
    this->~StableVector();
    new (this) StableVector(other);
}

template<typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::push(const ElementType& elem)
{
    cout << "--- 1c" << endl;
    m_elements.emplace_back();
    cout << "--- 2c" << endl;
    m_elements.back().set(elem);
    cout << "--- 3c" << endl;
    m_deleted.push_back(false);
    cout << "--- 4c" << endl;
    ++m_usedCount;
    return HandleT(size() - 1);
}

template<typename HandleT, typename ElemT>
HandleT StableVector<HandleT, ElemT>::push(ElementType&& elem)
{
    cout << "--- 1m" << endl;
    m_elements.emplace_back();
    cout << "--- 2m" << endl;
    m_elements.back().set(move(elem));
    cout << "--- 3m" << endl;
    m_deleted.push_back(false);
    cout << "--- 4m" << endl;
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

    m_deleted.resize(upTo.idx(), true);
    m_elements.resize(upTo.idx(), UnsafeWrapper());
}

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::increaseSize(HandleType upTo, const ElementType& elem)
{
    if (upTo.idx() < size())
    {
        panic("call to increaseSize() with a valid handle!");
    }

    auto sizeBefore = size();
    m_deleted.resize(upTo.idx(), false);
    m_elements.resize(upTo.idx(), UnsafeWrapper());
    for (size_t i = sizeBefore; i < upTo.idx(); i++)
    {
        m_elements[i].set(elem);
    }
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

    m_deleted[handle.idx()] = true;
    m_elements[handle.idx()].data.~ElementType();
    --m_usedCount;
}

template<typename HandleT, typename ElemT>
boost::optional<ElemT&> StableVector<HandleT, ElemT>::get(HandleType handle)
{
    if (handle.idx() >= size() || m_deleted[handle.idx()])
    {
        return boost::none;
    }
    return m_elements[handle.idx()].data;
}

template<typename HandleT, typename ElemT>
boost::optional<const ElemT&> StableVector<HandleT, ElemT>::get(HandleType handle) const
{
    if (handle.idx() >= size() || m_deleted[handle.idx()])
    {
        return boost::none;
    }
    return m_elements[handle.idx()].data;
}

template<typename HandleT, typename ElemT>
ElemT& StableVector<HandleT, ElemT>::operator[](HandleType handle)
{
    checkAccess(handle);
    return m_elements[handle.idx()].data;
}

template<typename HandleT, typename ElemT>
const ElemT& StableVector<HandleT, ElemT>::operator[](HandleType handle) const
{
    checkAccess(handle);
    return m_elements[handle.idx()].data;
}

template<typename HandleT, typename ElemT>
size_t StableVector<HandleT, ElemT>::size() const
{
    return m_deleted.size();
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
    if (m_deleted[handle.idx()])
    {
        ++m_usedCount;
        m_elements[handle.idx()].set(elem);
    }
    else
    {
        m_elements[handle.idx()].data = elem;
        m_deleted[handle.idx()] = false;
    }
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
    if (m_deleted[handle.idx()])
    {
        ++m_usedCount;
        m_elements[handle.idx()].set(elem);
    }
    else
    {
        m_elements[handle.idx()].data = elem;
        m_deleted[handle.idx()] = false;
    }
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

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
 * StableVector.tcc
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include "lvr2/util/Panic.hpp"
#include <boost/shared_array.hpp>

#include <sstream>
#include <string>


namespace lvr2
{

template<typename HandleT, typename ElemT>
void StableVector<HandleT, ElemT>::checkAccess(HandleType handle) const
{
    // Only actually check in debug mode, because checking this is costly...
    //#ifndef NDEBUG
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
    //#endif
}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(size_t countElements, const ElementType& defaultValue)
    : m_elements(countElements, defaultValue),
      m_usedCount(countElements)
{}

template<typename HandleT, typename ElemT>
StableVector<HandleT, ElemT>::StableVector(size_t countElements, const boost::shared_array<ElementType>& sharedArray)
    : m_usedCount(countElements)
{
    m_elements.reserve(countElements);
    #pragma omp parallel for
    for(size_t i=0; i<countElements; i++)
    {
        m_elements[i] = sharedArray[i];
    }
}

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
    const vector<boost::optional<ElemT>>* deleted,
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
bool StableVectorIterator<HandleT, ElemT>::isAtEnd() const
{
    return m_pos == m_elements->size();
}

template<typename HandleT, typename ElemT>
HandleT StableVectorIterator<HandleT, ElemT>::operator*() const
{
    return HandleT(m_pos);
}

} // namespace lvr2

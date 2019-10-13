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
 * VectorMap.tcc
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <utility>

#include "lvr2/util/Panic.hpp"

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
VectorMap<HandleT, ValueT>::VectorMap(size_t countElements, const boost::shared_array<ValueT>& sharedArray)
    : m_vec(countElements, sharedArray)
{}

template<typename HandleT, typename ValueT>
bool VectorMap<HandleT, ValueT>::containsKey(HandleT key) const
{
    return static_cast<bool>(m_vec.get(key));
}

template<typename HandleT, typename ValueT>
boost::optional<ValueT> VectorMap<HandleT, ValueT>::insert(HandleT key, const ValueT& value)
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
boost::optional<ValueT> VectorMap<HandleT, ValueT>::erase(HandleT key)
{
    auto val = m_vec.get(key);
    if (val)
    {
        auto out = ValueT(std::move(*val));
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
boost::optional<ValueT&> VectorMap<HandleT, ValueT>::get(HandleT key)
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
boost::optional<const ValueT&> VectorMap<HandleT, ValueT>::get(HandleT key) const
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

template<typename HandleT, typename ValueT>
std::unique_ptr<AttributeMapHandleIterator<HandleT>> VectorMapIterator<HandleT, ValueT>::clone() const
{
    return std::make_unique<VectorMapIterator>(*this);
}

} // namespace lvr2

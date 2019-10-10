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
 * AttributeMap.tcc
 *
 *  @date 26.07.2017
 */

#include "lvr2/util/Panic.hpp"

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

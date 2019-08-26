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
 * BaseHandle.tcc
 *
 *  @date 03.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */


#include <limits>

#include "lvr2/util/Panic.hpp"

using std::numeric_limits;

namespace lvr2
{


template <typename IdxT>
BaseHandle<IdxT>::BaseHandle(IdxT idx)
{
    setIdx(idx);
}

template <typename IdxT>
IdxT BaseHandle<IdxT>::idx() const
{
    return m_idx;
}
template <typename IdxT>
void BaseHandle<IdxT>::setIdx(IdxT idx)
{
    #ifndef NDEBUG
    if (idx == numeric_limits<IdxT>::max())
    {
        panic("Trying to create a Handle with MAX_INT as idx!");
    }
    #endif

    m_idx = idx;
}

template <typename IdxT>
bool BaseHandle<IdxT>::operator==(const BaseHandle& other) const
{
    return m_idx == other.m_idx;
}

template <typename IdxT>
bool BaseHandle<IdxT>::operator!=(const BaseHandle& other) const
{
    return m_idx != other.m_idx;
}

template <typename IdxT>
bool BaseHandle<IdxT>::operator<(const BaseHandle& other) const
{
    return m_idx < other.m_idx;
}

template <typename IdxT, typename NonOptionalT>
BaseOptionalHandle<IdxT, NonOptionalT>::BaseOptionalHandle()
    : m_idx(numeric_limits<IdxT>::max())
{}

template <typename IdxT, typename NonOptionalT>
BaseOptionalHandle<IdxT, NonOptionalT>::BaseOptionalHandle(BaseHandle<IdxT> handle)
    : m_idx(handle.idx())
{}

template <typename IdxT, typename NonOptionalT>
BaseOptionalHandle<IdxT, NonOptionalT>::BaseOptionalHandle(boost::optional<BaseHandle<IdxT>> handle)
    : m_idx(handle ? handle->idx() : numeric_limits<IdxT>::max())
{}

template <typename IdxT, typename NonOptionalT>
BaseOptionalHandle<IdxT, NonOptionalT>::BaseOptionalHandle(IdxT idx)
    : BaseOptionalHandle(BaseHandle<IdxT>(idx))
{}

template <typename IdxT, typename NonOptionalT>
BaseOptionalHandle<IdxT, NonOptionalT>::operator bool() const
{
    return m_idx != numeric_limits<IdxT>::max();
}

template <typename IdxT, typename NonOptionalT>
bool BaseOptionalHandle<IdxT, NonOptionalT>::operator!() const
{
    return m_idx == numeric_limits<IdxT>::max();
}

template <typename IdxT, typename NonOptionalT>
bool BaseOptionalHandle<IdxT, NonOptionalT>::operator==(const BaseOptionalHandle& other) const
{
    return m_idx == other.m_idx;
}

template <typename IdxT, typename NonOptionalT>
bool BaseOptionalHandle<IdxT, NonOptionalT>::operator!=(const BaseOptionalHandle& other) const
{
    return m_idx != other.m_idx;
}

template <typename IdxT, typename NonOptionalT>
NonOptionalT BaseOptionalHandle<IdxT, NonOptionalT>::unwrap() const
{
    return NonOptionalT(m_idx);
}

template <typename IdxT, typename NonOptionalT>
void BaseOptionalHandle<IdxT, NonOptionalT>::setIdx(IdxT idx)
{
    assert(idx != numeric_limits<IdxT>::max());
    m_idx = idx;
}

} // namespace lvr2

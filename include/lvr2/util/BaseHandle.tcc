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
 * BaseHandle.tcc
 *
 *  @date 03.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */


#include <limits>

#include <lvr2/util/Panic.hpp>

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
    if (idx == numeric_limits<IdxT>::max())
    {
        panic("Trying to create a Handle with MAX_INT as idx!");
    }

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
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
 * BaseHandle.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_BASEHANDLE_H_
#define LVR2_GEOMETRY_BASEHANDLE_H_

#include <boost/optional.hpp>

namespace lvr2
{

/**
 * @brief
 */
template<typename IdxT>
class BaseHandle
{
public:
    BaseHandle(IdxT idx);

    IdxT idx() const;
    void setIdx(IdxT idx);

    bool operator==(const BaseHandle& other) const;
    bool operator!=(const BaseHandle& other) const;

protected:
    IdxT m_idx;
};

template <typename IdxT, typename NonOptionalT>
class BaseOptionalHandle
{
public:
    BaseOptionalHandle();
    BaseOptionalHandle(BaseHandle<IdxT> handle);
    explicit BaseOptionalHandle(IdxT idx);

    explicit operator bool() const;

    bool operator!() const;

    bool operator==(const BaseOptionalHandle& other) const;
    bool operator!=(const BaseOptionalHandle& other) const;

    NonOptionalT unwrap() const;
    void setIdx(IdxT idx);

private:
    IdxT m_idx;
};

} // namespace lvr2

#include <lvr2/geometry/BaseHandle.tcc>

#endif /* LVR2_GEOMETRY_BASEHANDLE_H_ */

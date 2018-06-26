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

#ifndef LVR2_UTIL_BASEHANDLE_H_
#define LVR2_UTIL_BASEHANDLE_H_

#include <boost/optional.hpp>

namespace lvr2
{

/**
 * @brief Interface for all kinds of handles. Handles are basically a key to
 *        refer to something.
 *
 * From this class, a few concrete handle types (such as FaceHandle) will be
 * derived.
 *
 * Internally, the handle is just an index. How those indices are used is
 * determined by the thing creating handles (e.g. the mesh implementation).
 */
template<typename IdxT>
class BaseHandle
{
public:
    explicit BaseHandle(IdxT idx);

    IdxT idx() const;
    void setIdx(IdxT idx);

    bool operator==(const BaseHandle& other) const;
    bool operator!=(const BaseHandle& other) const;

protected:
    IdxT m_idx;
};

/**
 * @brief Base class for optional handles (handles that can be "null" or
 *        "None").
 *
 * This class is semantically equivalent to boost::optional<BaseHandle>. This
 * class uses a special index value to store the "None" value. This saves
 * memory.
 */
template <typename IdxT, typename NonOptionalT>
class BaseOptionalHandle
{
public:
    BaseOptionalHandle();
    BaseOptionalHandle(BaseHandle<IdxT> handle);
    BaseOptionalHandle(boost::optional<BaseHandle<IdxT>> handle);
    explicit BaseOptionalHandle(IdxT idx);

    explicit operator bool() const;

    bool operator!() const;

    bool operator==(const BaseOptionalHandle& other) const;
    bool operator!=(const BaseOptionalHandle& other) const;

    /**
     * @brief Extracts the handle. If `this` doesn't hold a handle (is "None"),
     *        this method panics.
     */
    NonOptionalT unwrap() const;
    void setIdx(IdxT idx);

private:
    IdxT m_idx;
};

} // namespace lvr2

#include <lvr2/util/BaseHandle.tcc>

#endif /* LVR2_UTIL_BASEHANDLE_H_ */

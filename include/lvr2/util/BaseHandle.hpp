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
 * BaseHandle.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 *  @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#ifndef LVR2_UTIL_BASEHANDLE_H_
#define LVR2_UTIL_BASEHANDLE_H_

#include <boost/optional.hpp>
#include <limits>

#include "lvr2/geometry/pmp/SurfaceMesh.h"
#include "lvr2/util/Panic.hpp"

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
using BaseHandle = pmp::Handle;

/**
 * @brief Base class for optional handles (handles that can be "null" or
 *        "None").
 *
 * This class is semantically equivalent to boost::optional<BaseHandle>. This
 * class uses a special index value to store the "None" value. This saves
 * memory.
 */
template <typename NonOptionalT = BaseHandle>
class BaseOptionalHandle : public BaseHandle
{
    pmp::IndexType idx() const = delete;

public:
    using BaseHandle::BaseHandle;
    BaseOptionalHandle()
        : BaseHandle()
    {}
    BaseOptionalHandle(NonOptionalT src)
        : BaseHandle(src)
    {}
    BaseOptionalHandle(boost::optional<BaseHandle> handle)
        : BaseHandle(handle ? handle->idx() : std::numeric_limits<pmp::IndexType>::max())
    {}

    explicit operator bool() const
    {
        return is_valid();
    }

    bool operator!() const
    {
        return !is_valid();
    }

    /**
     * @brief Extracts the handle. If `this` doesn't hold a handle (is "None"),
     *        this method panics.
     */
    NonOptionalT unwrap() const
    {
        if (!is_valid())
        {
            panic("Tried to unwrap invalid handle!");
        }
        return NonOptionalT(BaseHandle::idx());
    }
};

} // namespace lvr2

#endif /* LVR2_UTIL_BASEHANDLE_H_ */

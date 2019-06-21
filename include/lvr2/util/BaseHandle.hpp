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
template<typename IdxT=uint32_t >
class BaseHandle
{
public:
    explicit BaseHandle(IdxT idx);

    IdxT idx() const;
    void setIdx(IdxT idx);

    bool operator==(const BaseHandle& other) const;
    bool operator!=(const BaseHandle& other) const;
    bool operator<(const BaseHandle& other) const;

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

#include "lvr2/util/BaseHandle.tcc"

#endif /* LVR2_UTIL_BASEHANDLE_H_ */

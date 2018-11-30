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
 * Vector.tcc
 *
 *  @date 03.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include "Vector.hpp"

#include <lvr2/util/Panic.hpp>


namespace lvr2
{

template <typename BaseVecT>
Normal<BaseVecT>::Normal(BaseVecT base)
    : Vector<BaseVecT>(base)
{
    this->normalize();
}

template <typename BaseVecT>
Normal<BaseVecT>::Normal(Vector<BaseVecT> vec)
    : Vector<BaseVecT>(vec)
{
    this->normalize();
}

template <typename BaseVecT>
Normal<BaseVecT>::Normal(
    typename BaseVecT::CoordType x,
    typename BaseVecT::CoordType y,
    typename BaseVecT::CoordType z
)
    : Vector<BaseVecT>(x, y, z)
{
    this->normalize();
}

template<typename BaseVecT>
template<typename CollectionT>
Normal<BaseVecT>& Normal<BaseVecT>::average(const CollectionT& normals)
{
    if (normals.empty())
    {
        panic("average() of 0 normals");
    }

    Vector<BaseVecT> acc(0, 0, 0);
    for (auto n: normals)
    {
        static_assert(
            std::is_same<decltype(n), Normal<BaseVecT>>::value,
            "Collection has to contain Vectors"
        );
        acc += n.asVector();
    }
    return acc.normalized();
}

template <typename BaseVecT>
Normal<BaseVecT>& Normal<BaseVecT>::operator=(const Vector<BaseVecT>& other)
{
    if(&other != this)
    {
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        this->normalize();
    }
    return *this;
}

template <typename BaseVecT>
Normal<BaseVecT> Normal<BaseVecT>::operator-() const
{
    return Normal(-this->x, -this->y, -this->z);
}

template <typename BaseVecT>
Normal<BaseVecT> Normal<BaseVecT>::operator+(const Vector<BaseVecT>& other) const
{
    return Normal(other.x + this->x, other.y + this->y, other.z + this->z);
}

template <typename BaseVecT>
Normal<BaseVecT> Normal<BaseVecT>::operator-(const Vector<BaseVecT>& other) const
{
    return Normal(other.x - this->x, other.y - this->y, other.z - this->z);
}


} // namespace lvr2

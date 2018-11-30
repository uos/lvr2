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
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#include <cassert>

namespace lvr2
{

template<typename BaseVecT>
template<typename CollectionT>
Vector<BaseVecT> Vector<BaseVecT>::average(const CollectionT& vecs)
{
    Vector<BaseVecT> acc(0, 0, 0);
    size_t count = 0;
    for (auto v: vecs)
    {
        static_assert(
            std::is_same<decltype(v), Vector<BaseVecT>>::value,
            "Collection has to contain Vectors"
        );
        acc += v;
        count += 1;
    }
    return acc / count;
}

template<typename BaseVecT>
template<typename CollectionT>
Vector<BaseVecT> Vector<BaseVecT>::centroid(const CollectionT& points)
{
    Vector<BaseVecT> acc(0, 0, 0);
    size_t count = 0;
    for (auto p: points)
    {
        static_assert(
            std::is_same<decltype(p), Vector<BaseVecT>>::value,
            "Type mismatch in centroid calculation."
        );
        acc += p;
        count += 1;
    }
    return Vector<BaseVecT>(0, 0, 0) + acc / count;
}

template <typename BaseVecT>
typename BaseVecT::CoordType Vector<BaseVecT>::distanceFrom(const Vector<BaseVecT> &other) const
{
    return (*this - other).length();
}

template <typename BaseVecT>
typename BaseVecT::CoordType Vector<BaseVecT>::squaredDistanceFrom(const Vector<BaseVecT> &other) const
{
    return (*this - other).length2();
}


template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::cross(const Vector<BaseVecT> &other) const
{
    return BaseVecT::cross(other);
}

template <typename BaseVecT>
Normal<BaseVecT> Vector<BaseVecT>::normalized() const
{
    return Normal<BaseVecT>(*this);
}

template <typename BaseVecT>
void Vector<BaseVecT>::normalize()
{
    // Check for invalid vector. This check can be disabled in release mode,
    // which will probably lead to +inf and -inf values. In the documentation
    // for this function we require the vector to not be the null-vector.
    // assert(!(this->x == 0 && this->y == 0 && this->z == 0));
    if(!(this->x == 0 && this->y == 0 && this->z == 0))
    {
        auto len = this->length();
        this->x /= len;
        this->y /= len;
        this->z /= len;
    }
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator+(const Vector<BaseVecT>& other) const
{
    return BaseVecT::operator+(other);
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator-(const Vector<BaseVecT>& other) const
{
    return BaseVecT::operator-(other);
}

template <typename BaseVecT>
Vector<BaseVecT>& Vector<BaseVecT>::operator+=(const Vector<BaseVecT>& other)
{
    return static_cast<Vector<BaseVecT>&>(BaseVecT::operator+=(other));
}

template <typename BaseVecT>
Vector<BaseVecT>& Vector<BaseVecT>::operator-=(const Vector<BaseVecT>& other)
{
    return static_cast<Vector<BaseVecT>&>(BaseVecT::operator-=(other));
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator*(const typename BaseVecT::CoordType &scale) const
{
    return static_cast<Vector<BaseVecT>>(BaseVecT::operator*(scale));
}

template <typename BaseVecT>
Vector<BaseVecT> Vector<BaseVecT>::operator/(const typename BaseVecT::CoordType &scale) const
{
    return static_cast<Vector<BaseVecT>>(BaseVecT::operator/(scale));
}

} // namespace lvr2

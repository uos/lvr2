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

#include <cstring>

namespace lvr2 {

template<typename T>
Channel<T>::Channel()
: m_elementWidth(0)
, m_numElements(0)
{}

template<typename T>
Channel<T>::Channel(size_t n, size_t width)
: m_elementWidth(width), m_numElements(n)
, m_data(new T[n * width])
{}

template<typename T>
Channel<T>::Channel(size_t n, size_t width, DataPtr ptr)
: m_numElements(n)
, m_elementWidth(width)
, m_data(ptr)
{}

template<typename T>
Channel<T> Channel<T>::clone() const
{
    Channel<T> ret(m_numElements, m_elementWidth);
    std::memcpy(
            ret.dataPtr().get(),
            m_data.get(),
            sizeof(T) * m_numElements * m_elementWidth
        );
    return ret;
}

template<typename T>
ElementProxy<T> Channel<T>::operator[](const unsigned& idx)
{
    T* ptr = m_data.get();
    return ElementProxy<T>(&(ptr[idx * m_elementWidth]), m_elementWidth);
}

template<typename T>
const ElementProxy<T> Channel<T>::operator[](const unsigned& idx) const
{
    T* ptr = m_data.get();
    return ElementProxy<T>(&(ptr[idx * m_elementWidth]), m_elementWidth);
}

template<typename T>
size_t Channel<T>::width() const 
{
    return m_elementWidth;
}

template<typename T>
size_t Channel<T>::numElements() const 
{
    return m_numElements;
}

template<typename T>
const typename Channel<T>::DataPtr Channel<T>::dataPtr() const {
    return m_data;
}

template<typename T>
typename Channel<T>::DataPtr Channel<T>::dataPtr() {
    return m_data;
}


} // namespace lvr2
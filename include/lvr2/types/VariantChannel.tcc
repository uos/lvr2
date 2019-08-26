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


namespace lvr2 {

template<typename... T>
size_t VariantChannel<T...>::numElements() const
{
    return boost::apply_visitor(NumElementsVisitor(), *this);
}

template<typename... T>
size_t VariantChannel<T...>::width() const
{
    return boost::apply_visitor(WidthVisitor(), *this);
}

template<typename... T>
template<typename U>
boost::shared_array<U> VariantChannel<T...>::dataPtr() const
{
    return boost::apply_visitor(DataPtrVisitor<U>(), *this);
}

template<typename... T>
int VariantChannel<T...>::type() const
{
    return this->which();
}

template<typename... T>
template<typename U>
bool VariantChannel<T...>::is_type() const {
    return this->which() == index_of_type<U>::value;
}

template<typename... T>
template<typename U>
Channel<U> VariantChannel<T...>::extract() const
{
    return boost::get<Channel<U> >(*this);
}

template<typename... T>
template<typename U>
Channel<U>& VariantChannel<T...>::extract()
{
    return boost::get<Channel<U> >(*this);
}

template<typename... T>
VariantChannel<T...> VariantChannel<T...>::clone() const
{
    return boost::apply_visitor(CloneVisitor(), *this);
}

}
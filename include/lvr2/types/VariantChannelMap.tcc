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
template<typename U>
void VariantChannelMap<T...>::add(const std::string& name, Channel<U> channel)
{
    this->insert({name, channel});
}


template<typename... T>
template<typename U>
void VariantChannelMap<T...>::add(const std::string& name)
{
    this->insert({name, Channel<U>(0,0)});
}

template<typename... T>
template<typename U>
void VariantChannelMap<T...>::add(const std::string& name, size_t numElements, size_t width)
{
    this->insert({name, Channel<U>(numElements, width)});
}

template<typename... T>
template<typename U>
Channel<U>& VariantChannelMap<T...>::get(const std::string& name)
{
    return boost::get<Channel<U> >(this->at(name));
}

template<typename... T>
template<typename U>
const Channel<U>& VariantChannelMap<T...>::get(const std::string& name) const
{
    return boost::get<Channel<U> >(this->at(name));
}

template<typename... T>
template<typename U>
typename Channel<U>::Optional VariantChannelMap<T...>::getOptional(const std::string& name)
{
    typename Channel<U>::Optional ret;
    auto it = this->find(name);
    if(it != this->end() && it->second.template is_type<U>())
    {
        ret = boost::get<Channel<U> >(it->second);
    }

    return ret;
}

template<typename... T>
template<typename U>
const typename Channel<U>::Optional VariantChannelMap<T...>::getOptional(const std::string& name) const
{
    typename Channel<U>::Optional ret;
    auto it = this->find(name);
    if(it != this->end() && it->second.template is_type<U>())
    {
        ret = boost::get<Channel<U> >(it->second);
    }

    return ret;
}

template<typename... T>
int VariantChannelMap<T...>::type(const std::string& name) const
{
    return this->at(name).which();
}

template<typename... T>
template<typename U>
bool VariantChannelMap<T...>::is_type(const std::string& name) const
{
    return this->type(name) == index_of_type<U>::value;
}

template<typename... T>
template<typename U>
std::vector<std::string> VariantChannelMap<T...>::keys()
{
    std::vector<std::string> ret;

    for(auto it = this->typedBegin<U>(); it != this->end(); ++it)
    {
        ret.push_back(it->first);
    }

    return ret;
}

template<typename... T>
template<typename U>
size_t VariantChannelMap<T...>::numChannels()
{
    size_t ret = 0;

    for(auto it = this->typedBegin<U>(); it != this->end(); ++it)
    {
        ret ++;
    }

    return ret;
}

template<typename... T>
VariantChannelMap<T...> VariantChannelMap<T...>::clone() const
{
    VariantChannelMap<T...> ret;

    for(auto elem : *this)
    {
        ret.insert({elem.first, elem.second.clone()});
    }

    return ret;
}


} // namespace lvr2

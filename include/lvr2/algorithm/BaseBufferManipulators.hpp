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

#pragma once

#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <memory>
#include <boost/core/typeinfo.hpp>

#include "lvr2/types/MultiChannelMap.hpp"

namespace lvr2 {

namespace manipulators {

class Slice : public boost::static_visitor< MultiChannelMap::val_type > 
{
public:
    Slice(size_t left, size_t right)
    :m_left(left)
    ,m_right(right)
    {}

    template<typename T>
    MultiChannelMap::val_type operator()(Channel<T>& channel) const
    {
        MultiChannelMap::val_type vres;

        const size_t range = m_right - m_left;
        if(range < channel.numElements())
        {
            Channel<T> ret(range, channel.width());
            for(size_t i = m_left; i<m_right; i++)
            {
                for(int j=0; j<channel.width(); j++)
                {
                    ret[i-m_left][j] = channel[i][j];
                }
            }
            vres = ret;
        } else {
            vres = channel;
        }

        return vres;
    }
private:
    const size_t m_left;
    const size_t m_right;
};

template<typename T>
inline void no_delete(T* x)
{

}

class SliceShallow : public boost::static_visitor< MultiChannelMap::val_type > 
{
public:
    SliceShallow(size_t left, size_t right)
    :m_left(left)
    ,m_right(right)
    {}

    template<typename T>
    MultiChannelMap::val_type operator()(Channel<T>& channel) const
    {
        MultiChannelMap::val_type vres;

        const size_t range = m_right - m_left;

        size_t offset = m_left * channel.width();

        boost::core::typeinfo const & ti = BOOST_CORE_TYPEID(T);

        boost::shared_array<T> shallow_ptr(
            channel.dataPtr().get() + offset,
            no_delete<T>
        );

        Channel<T> ret(
            range,
            channel.width(),
            shallow_ptr
        );
        
        vres = ret;
        return vres;
    }
private:
    const size_t m_left;
    const size_t m_right;
};

class RandomSample : public boost::static_visitor< MultiChannelMap::val_type > 
{
public:
    RandomSample(size_t num_samples)
    :m_num_samples(num_samples)
    ,m_seed(std::chrono::steady_clock::now().time_since_epoch().count())
    {
    }

    template<typename T>
    MultiChannelMap::val_type operator()(Channel<T>& channel) const
    {
        MultiChannelMap::val_type vres;

        Channel<T> res(m_num_samples, channel.width());

        if(m_num_samples > channel.numElements() / 2)
        {
            std::vector<size_t> indices(channel.numElements());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(m_seed));
            
            for(size_t i=0; i<m_num_samples; i++)
            {
                for(size_t j=0; j<channel.width(); j++)
                {
                    res[i][j] = channel[indices[i]][j];
                }
            }
        } else {
            std::unordered_set<size_t> indices;
            std::srand(m_seed);
            auto eng = std::default_random_engine(m_seed);
            std::uniform_int_distribution<> distr(0, channel.numElements() - 1);

            while(indices.size() < m_num_samples )
            {
                indices.insert(distr(eng));
            }

            size_t i = 0;
            for(const size_t& index : indices)
            {
                for(size_t j=0; j<channel.width(); j++)
                {
                    res[i][j] = channel[index][j];
                }
                i++;
            }
        }

        vres = res;
        return vres;
    }

private:
    const size_t m_num_samples;
    const size_t m_seed;
};

} // namespace manipulators

} // namespace lvr2
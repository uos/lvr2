#pragma once

#include <random>
#include <chrono>
#include <algorithm>

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

        vres = res;

        return vres;
    }

private:
    const size_t m_num_samples;
    const size_t m_seed;
};

} // namespace manipulators

} // namespace lvr2
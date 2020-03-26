#ifndef LVR2_ASCII_RENDERER_BITFIELD_HPP
#define LVR2_ASCII_RENDERER_BITFIELD_HPP

#include <iostream>

namespace lvr2 {

template<unsigned int W, unsigned int H>
struct BitField {
    // storage
    unsigned char data = 0;

    BitField():data(0){}

    void set(const unsigned int& i, const unsigned int& j)
    {
        data |= 1UL << (i * H + j);
    }

    bool get(const unsigned int& i, const unsigned int& j) const
    {
        return (data >> (i * H + j)) & 1U;
    }

    void toggle(const unsigned int& i, const unsigned int& j)
    {
        data ^= 1UL << (i * H + j);
    }

    void clear(const unsigned int& i, const unsigned int& j)
    {
        data &= ~(1UL << (i * H + j) );
    }

    
};

} // namespace lvr2

#endif // LVR2_ASCII_RENDERER_BITFIELD_HPP
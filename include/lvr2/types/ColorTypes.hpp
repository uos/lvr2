#ifndef LVR2_TYPES_COLORTYPES_HPP
#define LVR2_TYPES_COLORTYPES_HPP

#include <cstdint>
#include <array>

namespace lvr2
{

using RGB8Color = std::array<uint8_t, 3>;
using RGBFColor = std::array<float, 3>;

} // namespace lvr2

#endif // LVR2_TYPES_COLORTYPES_HPP

#pragma once

#include <vector>

namespace lvr2
{
namespace meshio
{

typedef struct ArrayMetaStruct
{
    std::vector<size_t> shape;
    std::string data_type;
    static constexpr char entity[] = "channel";
    static constexpr char type[]   = "array";

} ArrayMeta;

} // namespace meshio
} // namespace lvr2
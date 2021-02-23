#include "lvr2/io/scanio/yaml/Matrix.hpp"

namespace YAML
{

bool isMatrix(const Node& node)
{
    if(!node["rows"] || !node["cols"] ||  !node["data"])
    {
        return false;
    }
    return true;
}

}
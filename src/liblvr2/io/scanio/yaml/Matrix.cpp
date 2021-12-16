#include "lvr2/io/scanio/yaml/Matrix.hpp"

namespace YAML
{

bool isMatrix(const Node& node)
{
    if (!node.IsSequence())
    {
        return false;
    }

    // check first entry
    YAML::const_iterator row_it = node.begin();
    
    if(!row_it->IsSequence())
    {
        return false;
    }

    return true;
}

}
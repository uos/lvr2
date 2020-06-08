#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

std::pair<std::string, std::string> getNames(
    const std::string& defaultGroup, 
    const std::string& defaultContainer, 
    const Description& d)
{
    std::string returnGroup = defaultGroup;
    std::string returnContainer = defaultContainer;

    if(d.groupName)
    {
        returnGroup = *d.groupName;
    }

    if(d.dataSetName)
    {
        returnContainer = *d.dataSetName;
    }

    return std::make_pair(returnGroup, returnContainer);
}

} // namespace lvr2
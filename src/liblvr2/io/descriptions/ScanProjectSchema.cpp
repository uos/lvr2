#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

std::ostream& operator<<(std::ostream& os, const Description& desc)
{
    os << "LVR Description: \n";
    if(desc.groupName)
    {
        os << "-- groupName: " << *desc.groupName << "\n";
    }

    if(desc.dataSetName)
    {
        os << "-- dataSetName: " << *desc.dataSetName << "\n";
    }

    if(desc.metaName)
    {
        os << "-- metaName: " << *desc.metaName << "\n";
    }
    
    return os;
}


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
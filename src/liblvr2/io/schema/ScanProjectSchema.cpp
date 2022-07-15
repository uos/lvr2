#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

std::ostream& operator<<(std::ostream& os, const Description& desc)
{
    os << "LVR Description: \n";
    if(desc.dataRoot)
    {
        os << "-- dataRoot: " << *desc.dataRoot << "\n";
    }

    if(desc.data)
    {
        os << "-- data: " << *desc.data << "\n";
    }

    if(desc.metaRoot)
    {
        os << "-- metaRoot: " << *desc.metaRoot << "\n";
    }

    if(desc.meta)
    {
        os << "-- meta: " << *desc.meta << "\n";
    }
    
    return os;
}

} // namespace lvr2
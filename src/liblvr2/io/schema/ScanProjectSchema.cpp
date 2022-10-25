#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/util/Timestamp.hpp"

namespace lvr2
{

std::ostream& operator<<(std::ostream& os, const Description& desc)
{
    os << timestamp << "[LVR Description]  -------------------------------------------------------" << std::endl;
    if(desc.dataRoot)
    {
        os << timestamp << "[LVR Description] DataRoot: " << *desc.dataRoot << std::endl;
    }

    if(desc.data)
    {
        os << timestamp << "[LVR Description] Data: " << *desc.data << std::endl;
    }

    if(desc.metaRoot)
    {
        os << timestamp << "[LVR Description] Meta Root: " << *desc.metaRoot << std::endl;
    }

    if(desc.meta)
    {
        os << timestamp << "[LVR Description] Meta: " << *desc.meta << std::endl;
    }
    
    return os;
}

} // namespace lvr2
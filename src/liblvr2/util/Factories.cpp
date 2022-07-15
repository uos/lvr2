#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaSlam6D.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaHDF5.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaHDF5V2.hpp"
#include <boost/algorithm/string.hpp>

namespace lvr2
{

DirectorySchemaPtr directorySchemaFromName(const std::string& schemaName, const std::string& rootDir)
{
    std::string name = boost::to_upper_copy(schemaName);
    if(name == "SLAM6D")
    {
        return DirectorySchemaPtr(new ScanProjectSchemaSlam6D(rootDir));
    }
    else if(name == "RAW")
    {
        return DirectorySchemaPtr(new ScanProjectSchemaRaw(rootDir));
    }
    else if(name == "RAW_PLY")
    {
        return DirectorySchemaPtr(new ScanProjectSchemaRawPly(rootDir));
    }
    std::cout << timestamp << "Unknown directory schema identifier: '" << name << "'." << std::endl;
    return nullptr;
}

HDF5SchemaPtr hdf5SchemafromName(const std::string& schemaName)
{
    std::string name = boost::to_upper_copy(schemaName);

    if(name == "HDF5")
    {
        return HDF5SchemaPtr(new ScanProjectSchemaHDF5);
    }
    else if(name == "HDF5V2")
    {
        return HDF5SchemaPtr(new ScanProjectSchemaHDF5);
    }
    std::cout << timestamp << "Unknown HDF5 schema identifier: '" << name << "'." << std::endl;
    return nullptr;
}



} // namespace lvr2
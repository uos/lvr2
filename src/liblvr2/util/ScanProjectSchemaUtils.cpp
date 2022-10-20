#include "lvr2/util/ScanProjectUtils.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/schema/ScanProjectSchemaEuRoC.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/schema/ScanProjectSchemaSlam6D.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5V2.hpp"

#ifdef LVR2_USE_RDB
#include "lvr2/io/schema/ScanProjectSchemaRdbx.hpp"
#endif

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace lvr2
{

DirectorySchemaPtr directorySchemaFromName(const std::string& schemaName, const std::string& rootDirectory)
{
    // Check root directory
    if(!boost::filesystem::is_directory(boost::filesystem::path(rootDirectory)))
    {
        std::cout << timestamp << "Cannot create directory schema. Given root is not a directory: '"
                  << rootDirectory << "'." << std::endl;
        return nullptr;
    }
    
    std::string name = boost::to_upper_copy<std::string>(schemaName);

    if(name == "EUROC")
    {
        std::cout << timestamp << "Creating ScanProjectSchemaEuRoC with root directory '" 
                  << rootDirectory << "." << std::endl;
        return DirectorySchemaPtr(new ScanProjectSchemaEuRoC(rootDirectory));
    } 
    else if (name == "HYPERLIB")
    {
        std::cout << timestamp << "Error: ScanProjectSchemaHyperlib not implemented yet." << std::endl;
        //return DirectorySchemaPtr(new ScanProjectSchemaHyperlib(rootDirectory));
        return nullptr;
    } 
    else if (name == "RAW")
    {
        std::cout << timestamp << "Creating ScanProjectSchemaRaw with root directory '" 
                  << rootDirectory << "." << std::endl;
        return DirectorySchemaPtr(new ScanProjectSchemaRaw(rootDirectory));
    }
      else if (name == "RAWPLY")
    {
        std::cout << timestamp << "Creating ScanProjectSchemaRawPly with root directory '" 
                  << rootDirectory << "." << std::endl;
        return DirectorySchemaPtr(new ScanProjectSchemaRaw(rootDirectory));
    }
    else if (name == "SLAM6D")
    {   
        std::cout << timestamp << "Creating ScanProjectSchemaSlam6D with root directory '" 
                  << rootDirectory << "." << std::endl;
        return DirectorySchemaPtr(new ScanProjectSchemaSlam6D(rootDirectory));
    }
#ifdef LVR2_USE_RDB
    else if (name == "RDBX")
    {
        std::cout << timestamp << "Creating ScanProjectSchemaRDBX with root directory '" 
                  << rootDirectory << "." << std::endl;
        return DirectorySchemaPtr(new ScanProjectSchemaRdbx(rootDirectory));
    }
#endif

    std::cout << timestamp << "Unknown directory schema name '" << schemaName << "'." << std::endl;
    return nullptr;
}

HDF5SchemaPtr hdf5SchemaFromName(const std::string& schemaName)
{
    std::string name = boost::to_upper_copy<std::string>(schemaName);

    if(name == "HDF5")
    {
        std::cout << timestamp << "Creating ScanProjectSchemaHDF5." << std::endl;
        return HDF5SchemaPtr(new ScanProjectSchemaHDF5);
    }
    else if(name == "HDFV5V2")
    {
        std::cout << timestamp << "Error: ScanProjectSchemaHDF5V2 not fully implemented." << std::endl;
        return nullptr;
        //std::cout << timestamp << "Creating ScanProjectSchemaHDF5V2." << std::endl;
        //return HDF5SchemaPtr(new ScanProjectSchemaHDF5V2);
    }

    std::cout << timestamp << "Unknown HDF5 schema name '" << schemaName << "'." << std::endl;
    return nullptr;
}

ScanProjectSchemaPtr schemaFromName(const std::string& schemaName, const std::string root)
{
    boost::filesystem::path path(root);

    if(boost::filesystem::is_directory(path))
    {
        return directorySchemaFromName(schemaName, root);
    }
    else if(boost::filesystem::extension(path) == ".h5")
    {
        return hdf5SchemaFromName(schemaName);
    }
    else
    {
        return nullptr;
    }
}

} //namespace lvr2
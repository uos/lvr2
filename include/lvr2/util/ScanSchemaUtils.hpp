#ifndef SCANSCHEMAUTILS
#define SCANSCHEMAUTILS

#include <string>
#include <vector>

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

/**
 * @brief Creates a directory schema from the given name
 * 
 * @param schemaName            A vaild directory schema name.
 * @param rootDir               The root directory for this schema
 * @return DirectorySchemaPtr   A pointer to a directory schema or nullptr 
 *                              if the schema name is unknown.
 */
DirectorySchemaPtr directorySchemaFromName(const std::string& schemaName, const std::string& rootDir);

/**
 * @brief Creates a HDF5 schema from given name 
 * 
 * @param schemaName            A vaild directory schema name. 
 * @return HDF5SchemaPtr        A pointer to a directory schema or nullptr 
 *                              if the schema name is unknown.
 */
HDF5SchemaPtr hdf5SchemaFromName(const std::string& schemaName);

/**
 * @brief A list of the implemented directory schema names
 */
const std::vector<std::string> implementedDirectorySchemas = {
    "EUROC", 
    "HYPERLIB", 
    "RAW", 
    "RAWPLY",
#ifdef LVR2_USE_RDB 
    "RDBX",
#endif
    "SLAM6D"};

/**
 * @brief A list of the implemented HDF5 schemas
 */
const std::vector<std::string> implementedHDF5Schemas = {"HDF5", "HDF5V2"};

} // namespace lvr2

#endif // SCANSCHEMAUTILS

#ifndef SCANPROJECTHELPER
#define SCANPROJECTHELPER

#include <utility>

#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

/**
 * @brief Extracts a scan from a scan project. 
 * 
 * @param project                           A pointer to a scan project
 * @param scanPositionNo                    The scan position number
 * @param lidarNo                           The lidar number
 * @param scanNo                            The scan for the given lidar
 * @param transform                         If true, the scan data will be transformed into 
 *                                          the scan projects reference system. Otherwise, the
 *                                          raw input data with the transformation with respect to
 *                                          the corresponsing scan position will be returned.
 * 
 * @return std::pair<ScanPtr, Transformd>   A pair of scan data and transformation into the global
 *                                          reference frame. If the scan data could not be loaded,
 *                                          the ScanPtr is a nullptr and the corresponding transformation
 *                                          an identity matrix.
 */
std::pair<ScanPtr, Transformd> scanFromProject(
    ScanProjectPtr project, 
    size_t scanPositionNo, size_t lidarNo = 0, size_t scanNo = 0);


/**
 * @brief Loads a scanProject from given file / directory
 * 
 * @param file                              A HDF5-File containing a scan project
 * @param schemaName                        The name of a known HDF5 schema
 * @return ScanProjectPtr 
 */
ScanProjectPtr scanProjectFromHDF5(std::string file, const std::string& schemaName);

/**
 * @brief Creates a scan project representation from a single file
 * 
 * @param file                              A file with point cloud data
 * @return ScanProjectPtr                   A new scan project or nullptr if loading failed
 */
ScanProjectPtr scanProjectFromFile(const std::string& file);

/**
 * @brief Creates a scan project from a directory containing .ply files. The function
 *        assumes that all .ply files in the directory contain point cloud data
 *        and that all point clouds are already registered in a common coordinate system.
 * 
 * @param dir                               A directory containing .ply files with point
 *                                          cloud data
 * @return ScanProjectPtr                   A new scan project or nullpt if loading failed.
 */
ScanProjectPtr scanProjectFromPLYFiles(const std::string& dir);

} // namespace LVR2

#endif // SCANPROJECTHELPER
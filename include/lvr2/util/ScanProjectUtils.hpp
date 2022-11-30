#ifndef SCANPROJECTUTILS
#define SCANPROJECTUTILS
 
#undef USE_UNORDERED_MAP

#include <utility>
#include <boost/optional.hpp>

#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
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
 * @param file          A HDF5-File containing a scan project
 * @param schemaName    The name of a known HDF5 schema
 * @return ScanProjectPtr 
 */
ScanProjectPtr scanProjectFromHDF5(std::string file, const std::string& schemaName);

/**
 * @brief Creates a scan project representation from a single file
 * 
 * @param file      A file with point cloud data
 * @return ScanProjectPtr   A new scan project or nullptr if loading failed
 */
ScanProjectPtr scanProjectFromFile(const std::string& file);

/**
 * @brief Creates a scan project from a directory containing .ply files. The function
 *        assumes that all .ply files in the directory contain point cloud data
 *        and that all point clouds are already registered in a common coordinate system.
 * 
 * @param dir       A directory containing .ply files with point cloud data
 * @return          A new scan project or nullptr if loading failed.
 */
ScanProjectPtr scanProjectFromPLYFiles(const std::string& dir);

/**
 * @brief Loads a scan project from given source. 
 * 
 * @param schema    The schema name that describes the project's structure 
 * @param source    The data source
 * @param loadData  If true, all data will be loaded. If false, only meta
 *                  data and the project structure will be parsed.
 * 
 * @return ScanProjectPtr   A new scan project or nullptr if loading failed.
 */
ScanProjectPtr loadScanProject(
    const std::string& schema, 
    const std::string& source, 
    bool loadData = false);

/**
 * @brief Saves given scan project to a target using the given schema
 * 
 * @param project   A scan project
 * @param schema    The schema for the written data
 * @param target    The target (where to write, i.e., directory or HDF5 file)
 */
void saveScanProject(
    ScanProjectPtr& project,
    const std::string& schema, 
    const std::string& target);

/**
 * @brief Saves given scan positions of a scan project to a target 
 *        using the given schema.
 * 
 * @param project   Source scan project 
 * @param positions Scan positions that shall be saved
 * @param schema    Schema of the target
 * @param target    Target destination for saved data
 */
void saveScanProject(
    ScanProjectPtr& project,
    const std::vector<size_t>& positions,
    const std::string& schema,
    const std::string& target);

/**
 * @brief Retuns a new scan project that holds the given scan positions
 *        from the source project
 * 
 * @param p         The source project          
 * @param indices   Indices of scan positions that shall be included 
 *                  in the returned scan project
 */
ScanProjectPtr getSubProject(ScanProjectPtr p, std::vector<size_t> indices);

/**
 * @brief Prints infos about all entities in the given scan project
 */
void printScanProjectStructure(const ScanProjectPtr project);

/**
 * @brief Prints detailed info about the scan position
 */
void printScanPositionStructure(const ScanPositionPtr p);

/**
 * @brief Prints detailed infor about the LiDAR
 */
void printLIDARStructure(const LIDARPtr p);

/**
 * @brief Prints detailed infor about the camera
 */
void printCameraStructure(const CameraPtr p);

/**
 * @brief Prints detailed infor about the hyperspectral camera
 */
void printHyperspectralCameraStructure(const HyperspectralCameraPtr p);

/**
 * @brief Prints detailed infor about the scan
 */
void printScanStructure(const ScanPtr p);

/**
 * @brief Prints detailed infor about the camera image
 */
void printCameraImageStructure(const CameraImagePtr p);

/**
 * @brief Prints detailed infor about the camera image group
 */
void printCameraImageGroupStructure(const CameraImageGroupPtr p);

/**
 * @brief Prints detailed infor about the hyperspectral panorama
 */
void printHyperspectralPanoramaStructure(const HyperspectralPanoramaPtr p);

/**
 * @brief Estimates normals for each scan position in the 
 *        scan project. The results are written back to the
 *        source, so schema and kernel that were used to load
 *        the project have to support writing 
 * 
 * @param project   The scan project
 * @param kn        Number of nearest neighbors for normal estimation
 * @param ki        Number of nearest neighbors for normal interpolation
 */
void estimateProjectNormals(
    ScanProjectPtr project,
    size_t kn,
    size_t ki
);

/**
 * @brief Creates a scan project consisting of only the given 
 *        scan position indices
 * 
 * @param schema    A schema name
 * @param root      Root resource containing the data
 * @param positions Position indices that shall be loaded
 * @return ScanProjectPtr A new scan project
 */
ScanProjectPtr loadScanPositionsExplicitly(
    const std::string& schema,
    const std::string& root,
    const std::vector<size_t>& positions);

/**
 * @brief Writes the point cloud data from a scan project to a 
 *        PLY file. It transforms all point cloud according to the 
 *        stored transformation into the global coordinate system.
 *        Exported are points, normals (if present) and colors (if present).   
 * 
 *        Normals and colors will only be exported if they exist for 
 *        all scans.  
 *     
 * @param project           A scan project
 * @param plyFile           The target .ply file
 * @param firstScanOnly     If true, only the first scan of each lidar will be 
 *                          exported, otherwise all scans will be loaded, converted
 *                          written to the target file.
 */
void exportScanProjectToPLY(ScanProjectPtr project, const std::string plyFile, bool firstScanOnly = true, PointReductionAlgorihmTag = NONE);



} // namespace LVR2

#endif // SCANPROJECTUTILS

#ifndef __SCANIOUTILS_HPP__
#define __SCANIOUTILS_HPP__

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PLYIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/registration/CameraModels.hpp"

namespace lvr2
{

/**
 * @brief       Gets the sensor type for a given directory. Return an
 *              empty string if the directory does not contain a meta.yaml
 *              file with 'sensor_type' element.
 *
 * @param dir   Directory to check for sensor data
 * @return      A string tag that identifies the directory content
 */
std::string getSensorType(const boost::filesystem::path& dir);


//////////////////////////////////////////////////////////////////////////////////
/// SCANIMAGE
//////////////////////////////////////////////////////////////////////////////////

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const size_t& positionNumber,
    const size_t& cameraNumber,
    const size_t& imageNumber);

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const std::string positionDirectory,
    const size_t& cameraNumber,
    const size_t& imageNumber);

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const std::string positionDirectory,
    const std::string cameraDirectory,
    const size_t& imageNr);

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const std::string& positionDirectory,
    const size_t& cameraNumber,
    const size_t& imageNumber);

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const size_t& positionNumber,
    const size_t& cameraNumber,
    const size_t& imageNumber);

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const std::string& positionDirectory,
    const std::string& cameraDirectory,
    const size_t& imageNumber);

void loadScanImages(
    vector<ScanImagePtr>& images,
    boost::filesystem::path dataPath);

//////////////////////////////////////////////////////////////////////////////////
/// SCANCAMERA
//////////////////////////////////////////////////////////////////////////////////

void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& image,
    const std::string positionDirectory,
    const std::string cameraDirectory);

void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& image,
    const size_t& positionNumber,
    const size_t& cameraNumber);

void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& image,
    const std::string& positionDirectory,
    const size_t& cameraNumber);

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& image,
    const std::string& positionDirectory,
    const std::string& cameraDirectory);

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& image,
    const std::string& positionDirectory,
    const size_t& cameraNumber);

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& image,
    const size_t& positionNumber,
    const size_t& cameraNumber);

//////////////////////////////////////////////////////////////////////////////////
/// SCAN
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Save a Scan struct.
 *
 * @param root                  Project root directory
 * @param scan                  The scan object to save
 * @param positionName          The name of the scan position
 * @param scanDirectoryName     The name of the scan directory
 * @param scanName              The name of the generated scan file
 */
void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const std::string positionName,
    const std::string scanDirectoryName,
    const std::string scanName);

void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const std::string positionDirectory,
    const std::string scanDirectory,
    const size_t& scanNumber);

void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const size_t& positionNumber,
    const size_t& scanNumber);

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const std::string& positionDirectory,
    const std::string& scanDirectory,
    const std::string& scanName);

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const std::string& positionDirectory,
    const std::string& scanDirectory,
    const size_t& scanNumber);

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const size_t& positionNumber,
    const size_t& scanNumber);


//////////////////////////////////////////////////////////////////////////////////
/// SCAN_POSITION
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Save a ScanPosition struct.
 *
 * @param root                  Project root directory
 * @param scanPos                  The scan object to save
 * @param positionName          The name of the scan position
 * @param scanDirectoryName     The name of the scan directory
 * @param scanName              The name of the generated scan file
 */
void saveScanPosition(
    const boost::filesystem::path& root,
    const ScanPosition& scanPos,
    const std::string positionDirectory);

void saveScanPosition(
    const boost::filesystem::path& root,
    const ScanPosition& scanPos,
    const size_t& positionNumber);

bool loadScanPosition(
    const boost::filesystem::path& root,
    ScanPosition& scanPos,
    const std::string& positionDirectory);

bool loadScanPosition(
    const boost::filesystem::path& root,
    ScanPosition& scanPos,
    const size_t& positionNumber);


//////////////////////////////////////////////////////////////////////////////////
/// SCAN_PROJECT
//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Save a ScanProject struct.
 *
 * @param root                  Project root directory
 * @param scanProj              The scanproject object to save
 */
void saveScanProject(
    const boost::filesystem::path& root,
    const ScanProject& scanProj);

bool loadScanProject(
    const boost::filesystem::path& root,
    ScanProject& scanProj);


// std::set<size_t> loadPositionIdsFromDirectory(
//     const boost::filesystem::path& path
// );

// std::set<size_t> loadCamIdsFromDirectory(
//     const boost::filesystem::path& path,
//     const size_t& positionNr
// );

// std::set<size_t> loadImageIdsFromDirectory(
//     const boost::filesystem::path& path,
//     const size_t& positionNr,
//     const size_t& camNr
// );

// void saveScanToDirectory(const boost::filesystem::path& path, const Scan& scan, const size_t& positionNr);

// bool loadScanFromDirectory(const boost::filesystem::path&, Scan& scan, const size_t& positionNr, bool loadData);

// void saveScanToHDF5(const std::string filename, const size_t& positionNr);

// bool loadScanFromHDF5(const std::string filename, const size_t& positionNr);

// void saveScanImageToDirectory(
//     const boost::filesystem::path& path,
//     const std::string& camDir,
//     const ScanImage& image,
//     const size_t& positionNr,
//     const size_t& imageNr);

// bool loadScanImageFromDirectory(
//     const boost::filesystem::path& path,
//     const std::string& camDir,
//     ScanImage& image,
//     const size_t& positionNr,
//     const size_t& imageNr);



// void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr);

// bool loadScanPositionFromDirectory(const boost::filesystem::path& path, ScanPosition& position, const size_t& positionNr);

// void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& position);

// bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& position);

// void writeScanMetaYAML(const boost::filesystem::path& path, const Scan& scan);

// void writeScanImageMetaYAML(const boost::filesystem::path& path, const ScanImage& image);

// void loadScanMetaInfoFromYAML(const boost::filesystem::path& path, Scan& scan);

// void loadImageMetaInfoFromYAML(const boost::filesystem::path&, ScanImage& image);

// void loadPinholeModelFromYAML(const boost::filesystem::path& path, PinholeModeld& model);

// void writePinholeModelToYAML(const boost::filesystem::path& path, const PinholeModeld& model);


} // namespace lvr2

#endif

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

#ifndef __SCANIOUTILS_HPP__
#define __SCANIOUTILS_HPP__

#include <string>

#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

void saveScanToDirectory(const boost::filesystem::path& path, const Scan& scan, const size_t& positionNr);

bool loadScanFromDirectory(const boost::filesystem::path&, Scan& scan, const size_t& positionNr);

void saveScanToHDF5(const std::string filename, const size_t& positionNr);

bool loadScanFromHDF5(const std::string filename, const size_t& positionNr);

void saveScanImageToDirectory(const boost::filesystem::path& path, const ScanImage& image, const size_t& positionNr);

bool loadScanImageFromDirectory(const boost::filesystem::path& path, ScanImage& image, const size_t& positionNr);

void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr);

bool loadScanPositionFromDirectory(const boost::filesystem::path& path, ScanPosition& position, const size_t& positionNr);

void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& position, const size_t& positionNr);

bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& position, const size_t& positionNr);


} // namespace lvr2 

#endif

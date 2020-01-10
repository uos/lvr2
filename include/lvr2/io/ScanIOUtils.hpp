#ifndef __SCANIOUTILS_HPP__
#define __SCANIOUTILS_HPP__

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PLYIO.hpp"
#include "lvr2/registration/PinholeCameraModel.hpp"

#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

namespace lvr2
{

void saveScanToDirectory(const boost::filesystem::path& path, const Scan& scan, const size_t& positionNr);

bool loadScanFromDirectory(const boost::filesystem::path&, Scan& scan, const size_t& positionNr, bool loadData);

void saveScanToHDF5(const std::string filename, const size_t& positionNr);

bool loadScanFromHDF5(const std::string filename, const size_t& positionNr);

void saveScanImageToDirectory(const boost::filesystem::path& path, const ScanImage& image, const size_t& positionNr, const size_t& imageNr);

bool loadScanImageFromDirectory(const boost::filesystem::path& path, ScanImage& image, const size_t& positionNr, const size_t& imageNr);

void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr);

bool loadScanPositionFromDirectory(const boost::filesystem::path& path, ScanPosition& position, const size_t& positionNr);

void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& position);

bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& position);

void writeScanMetaYAML(const boost::filesystem::path& path, const Scan& scan);

void writeScanImageMetaYAML(const boost::filesystem::path& path, const ScanImage& image);

void loadScanMetaInfoFromYAML(const boost::filesystem::path& path, Scan& scan);

void loadImageMetaInfoFromYAML(const boost::filesystem::path&, ScanImage& image);

void loadPinholeModelFromYAML(const boost::filesystem::path& path, PinholeCameraModeld& model);

void writePinholeModelToYAML(const boost::filesystem::path& path, const PinholeCameraModeld& model);

template<typename T, int Rows, int Cols>
Eigen::Matrix<T, Rows, Cols> loadMatrixFromYAML(const YAML::const_iterator& it);

template<typename T, int Rows, int Cols>
void saveMatrixToYAML(YAML::Node& node, const std::string& name, const Eigen::Matrix<T, Rows, Cols>& matrix);

template<typename T>
void loadArrayFromYAML(const YAML::const_iterator& it, T* array, size_t n);



} // namespace lvr2 

#include "ScanIOUtils.tcc"

#endif

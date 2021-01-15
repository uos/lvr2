#ifndef HDF5UTIL_HPP
#define HDF5UTIL_HPP

#pragma once

#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <H5Tpublic.h>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <chrono>
#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <vector>

namespace lvr2
{

namespace hdf5util
{

template<typename T>
void addAtomic(HighFive::Group& g,
    const std::string datasetName,
    const T data);

void addString(
    HighFive::Group& g,
    const std::string datasetName,
    const std::string data);

template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    std::vector<size_t>& dim, 
    boost::shared_array<T>& data);

template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    const size_t& length, 
    boost::shared_array<T>& data);

template<typename T>
void addVector(HighFive::Group& g,
    const std::string datasetName,
    const std::vector<T>& data);

template<typename T>
boost::optional<T> getAtomic(
    const HighFive::Group& g,
    const std::string datasetName);

boost::optional<std::string> getString(
    const HighFive::Group& g,
    const std::string& datasetName);

template<typename T>
boost::shared_array<T> getArray(
    const HighFive::Group& g, 
    const std::string& datasetName,
    std::vector<size_t>& dim);

template<typename T>
boost::shared_array<T> getArray(
    const HighFive::Group& g, 
    const std::string& datasetName,
    size_t& dim);

template<typename T>
boost::optional<std::vector<T> > getVector(
    const HighFive::Group& g, 
    const std::string& datasetName);

template<typename T>
std::vector<size_t> getDimensions(
    const HighFive::Group& g, 
    const std::string& datasetName);

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void addMatrix(HighFive::Group& group,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat);

template<typename MatrixT>
boost::optional<MatrixT> getMatrix(const HighFive::Group& g, const std::string& datasetName);

std::vector<std::string> splitGroupNames(const std::string& groupName);

std::pair<std::string, std::string> validateGroupDataset(
    const std::string& groupName, 
    const std::string& datasetName);

void writeBaseStructure(std::shared_ptr<HighFive::File> hdf5_file);

HighFive::Group getGroup(std::shared_ptr<HighFive::File> hdf5_file,
                                const std::string& groupName,
                                bool create = true);

HighFive::Group getGroup(HighFive::Group& g, const std::string& groupName, bool create = true);


bool exist(const std::shared_ptr<HighFive::File>& hdf5_file, const std::string& groupName);


bool exist(const HighFive::Group& group, const std::string& groupName);

std::shared_ptr<HighFive::File> open(const std::string& filename);

template <typename T>
std::unique_ptr<HighFive::DataSet> createDataset(HighFive::Group& g,
                                                 std::string datasetName,
                                                 const HighFive::DataSpace& dataSpace,
                                                 const HighFive::DataSetCreateProps& properties);

template <typename T>
void setAttribute(HighFive::Group& g, const std::string& attr_name, T& data);

// For DataSet as well
// template <typename T>
// void setAttribute(HighFive::DataSet& d, const std::string& attr_name, T& data);



} // namespace hdf5util

} // namespace lvr2

#include "Hdf5Util.tcc"

#endif

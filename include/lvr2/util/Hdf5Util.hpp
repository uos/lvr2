#ifndef HDF5UTIL_HPP
#define HDF5UTIL_HPP

#pragma once

// HDF5-related includes
#include <H5Tpublic.h>
#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

// Boost libraries
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

// C++ standard library
#include <chrono>
#include <memory>
#include <string>
#include <vector>

// LVR internal includes
#include "lvr2/util/Tuple.hpp"
#include "lvr2/io/scanio/yaml/Matrix.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{

namespace hdf5util
{

using H5AllowedTypes = Tuple<
        char,
        signed char,
        unsigned char,
        short,
        unsigned short,
        int, 
        unsigned,
        long,
        unsigned long,
        long long,
        unsigned long long,
        float,
        double,
        bool,
        std::string
    >;

/**
 * @brief Adds a atomic as dataset to a group
 * 
 * @tparam T Atomic type (float, int, std::string)
 * @param g 
 * @param datasetName 
 * @param data 
 */
template<typename T>
void addAtomic(HighFive::Group& g,
    const std::string datasetName,
    const T data);

/**
 * @brief Adds an array (of possibly multiple dimensions) as dataset to a group
 * 
 * @tparam T 
 * @param g 
 * @param datasetName 
 * @param dim 
 * @param data 
 */
template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    std::vector<size_t>& dim, 
    boost::shared_array<T>& data);

/**
 * @brief Adds a flat array as dataset to a group
 * 
 * @tparam T 
 * @param g 
 * @param datasetName 
 * @param length 
 * @param data 
 */
template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    const size_t& length, 
    boost::shared_array<T>& data);

/**
 * @brief Adds a std::vector of atomic type as dataset to a group
 * 
 * @tparam T 
 * @param g 
 * @param datasetName 
 * @param data 
 */
template<typename T>
void addVector(HighFive::Group& g,
    const std::string datasetName,
    const std::vector<T>& data);

/**
 * @brief Adds an Eigen::Matrix as dataset to a group 
 * 
 * @param group 
 * @param datasetName 
 * @param mat 
 */
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void addMatrix(HighFive::Group& group,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat);

/**
 * @brief Gets a 
 * 
 * @tparam T 
 * @param g 
 * @param datasetName 
 * @return boost::optional<T> 
 */
template<typename T>
boost::optional<T> getAtomic(
    const HighFive::Group& g,
    const std::string datasetName);

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


template<typename MatrixT>
boost::optional<MatrixT> getMatrix(const HighFive::Group& g, const std::string& datasetName);

std::vector<std::string> splitGroupNames(const std::string& groupName);

/**
 * @brief 
 * 
 * @param groupName groupName   like groupA/groupB
 * @param datasetName datasetName like groupC/dataset (group hides in dataset name)
 * @return std::pair<std::string, std::string> Returns corrected groupName and datasetName
 */
std::pair<std::string, std::string> validateGroupDataset(
    const std::string& groupName, 
    const std::string& datasetName);

/**
 * @brief Write Base structure to hdf5_file.
 * Base structure consists of
 * - version tag
 * 
 * @param hdf5_file 
 */
void writeBaseStructure(std::shared_ptr<HighFive::File> hdf5_file);

/**
 * @brief Get the Group from hdf5 file with specified groupName(string).
 *      
 * @param hdf5_file hdf5 file to search in
 * @param groupName groupName as std::string
 * @param create    if true: creates the group once it doesnt exist
 * @return HighFive::Group 
 */
HighFive::Group getGroup(std::shared_ptr<HighFive::File> hdf5_file,
                                const std::string& groupName,
                                bool create = true);

/**
 * @brief Get the Group from a Group with specified groupName(string).
 *      
 * @param g hdf5 group to search in
 * @param groupName groupName as std::string
 * @param create    if true: creates the group once it doesnt exist
 * @return HighFive::Group 
 */
HighFive::Group getGroup(HighFive::Group& g, const std::string& groupName, bool create = true);


struct HighFiveSplit
{
    std::vector<std::string> groups;
    std::vector<std::string> datasets;
};

/**
 * @brief Split objects of HDF5 Group into Groups/Datasets
 * 
 * @param g 
 * @return HighFiveSplit 
 */
HighFiveSplit split(HighFive::Group g);

/**
 * @brief Checks if a certain group name exists in a file
 * 
 * @param hdf5_file file
 * @param groupName name of the group as std::string. Can be nested group "groupA/groupB/groupC".
 * @return true   If group exists
 * @return false  If group doesnt exist
 */
bool exist(const std::shared_ptr<HighFive::File>& hdf5_file, 
            const std::string& groupName);

/**
 * @brief Checks if a certain group name exists in a group
 * 
 * @param group    Hdf5 Group 
 * @param groupName name of the group as std::string. Can be nested group "groupA/groupB/groupC".
 * @return true   If group exists
 * @return false  If group doesnt exist
 */
bool exist(const HighFive::Group& group, const std::string& groupName);

/**
 * @brief Open Helper function
 * 
 * @param filename   Path to Hdf5 file
 * @return std::shared_ptr<HighFive::File> shared_ptr of HighFive::File object 
 */
std::shared_ptr<HighFive::File> open(const std::string& filename, 
    unsigned int flag = HighFive::File::ReadWrite);

/**
 * @brief Create a Hdf5 Dataset savely. 
 * Special behaviors over the normal HighFive::Group.createDataset:
 * - If there is an existing dataset of same type and shape:
 *   -> return it instead of a new constructed
 * - If there is an existing dataset of same type but different shape: 
 *   -> try to resize the dataset and return the result
 * - If there is an existing dataset of different type
 *   -> need to delete the dataset
 * - Else:
 *   -> same behavior as HighFive::Group::createDataset
 * 
 * @tparam T 
 * @param g 
 * @param datasetName 
 * @param dataSpace 
 * @param properties 
 * @return std::unique_ptr<HighFive::DataSet> 
 */
template <typename T>
std::unique_ptr<HighFive::DataSet> createDataset(HighFive::Group& g,
                                                 std::string datasetName,
                                                 const HighFive::DataSpace& dataSpace,
                                                 const HighFive::DataSetCreateProps& properties);

/**
 * @brief Sets an atomic type as Group-Attribute
 * 
 * @tparam T 
 * @param g 
 * @param attr_name 
 * @param data 
 */
template <typename T, typename HT>
void setAttribute(
    HT& g,
    const std::string& attr_name,
    const T& data);

template<typename T, typename HT>
void setAttributeVector(
    HT& g,
    const std::string& attr_name,
    const std::vector<T>& vec
);

template<typename T, typename HT>
void setAttributeArray(
    HT& g,
    const std::string& attr_name,
    boost::shared_array<T> data,
    size_t size
);

template<typename HT>
void setAttributeMatrix(
    HT& g, 
    const std::string& attr_name,
    const Eigen::MatrixXd& mat);



/**
 * @brief Checks if an Group-Attributes value equals a atomic
 * 
 * @tparam T 
 * @param g 
 * @param attr_name 
 * @param data 
 * @return true 
 * @return false 
 */
template <typename HT, typename T>
bool checkAttribute(HT& g, const std::string& attr_name, T& data);

/**
 * @brief Get a Group-Attributes value
 * 
 * @tparam HT HighFive::Group or HighFive::Dataset
 * @tparam T 
 * @param g 
 * @param attr_name 
 * @return boost::optional<T> 
 */
template <typename T, typename HT>
boost::optional<T> getAttribute(
    const HT& g,
    const std::string& attr_name);

template<typename T, typename HT>
boost::optional<std::vector<T> > getAttributeVector(
    const HT& g,
    const std::string& attr_name);

template<typename HT>
boost::optional<Eigen::MatrixXd> getAttributeMatrix(
    const HT& g,
    const std::string& attr_name);

/**
 * @brief Converts HighFive::DataType.string() to the corresponding lvr2 Channel Type.
 * 
 * @param h5type 
 * @return boost::optional<std::string> 
 */
boost::optional<std::string> highFiveTypeToLvr(std::string h5type);


template<typename HT>
void setAttributeMeta(
    HT& g,
    YAML::Node node,
    std::string prefix=""
);


template<typename HT>
YAML::Node getAttributeMeta(
    const HT& g
);



} // namespace hdf5util

} // namespace lvr2

#include "Hdf5Util.tcc"

#endif

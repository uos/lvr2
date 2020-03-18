#pragma once
#ifndef LVR2_IO_HDF5_ARRAYIO_HPP
#define LVR2_IO_HDF5_ARRAYIO_HPP

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include <boost/shared_array.hpp>

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class ArrayIO {
public:

    template<typename T>
    boost::shared_array<T> loadArray(
        std::string groupName,
        std::string datasetName,
        size_t& size);

    template<typename T>
    boost::shared_array<T> load(
        std::string groupName,
        std::string datasetName,
        size_t& size);

    template<typename T>
    boost::shared_array<T> load(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dim);

    template<typename T>
    boost::shared_array<T> load(
        HighFive::Group& g,
        std::string datasetName,
        std::vector<size_t>& dim
    );

    template<typename T>
    void save(
        std::string groupName,
        std::string datasetName,
        size_t size,
        boost::shared_array<T> data);

    template<typename T>
    void save(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        boost::shared_array<T> data);

    template<typename T>
    void save(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        std::vector<hsize_t>& chunkSize,
        boost::shared_array<T> data);

    template<typename T>
    void save(HighFive::Group& g,
        std::string datasetName,
        std::vector<size_t>& dim,
        std::vector<hsize_t>& chunkSize,
        boost::shared_array<T>& data);

protected:
    Derived* m_file_access = static_cast<Derived*>(this);

};

} // namespace hdf5_features

} // namespace lvr2

#include "ArrayIO.tcc"

#endif

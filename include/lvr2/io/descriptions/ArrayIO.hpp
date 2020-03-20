#pragma once
#ifndef LVR2_IO_HDF5_ARRAYIO_HPP
#define LVR2_IO_HDF5_ARRAYIO_HPP

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include <boost/shared_array.hpp>

namespace lvr2 {

template<typename FeatureBase>
class ArrayIO {
public:

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

protected:
    FeatureBase* m_featureBase= static_cast<FeatureBase*>(this);

};

} // namespace lvr2

#include "ArrayIO.tcc"

#endif

#pragma once
#ifndef LVR2_IO_HDF5_ARRAYIO_HPP
#define LVR2_IO_HDF5_ARRAYIO_HPP

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include <boost/shared_array.hpp>

namespace lvr2 {

template<typename FeatureBase>
class ArrayIO {
public:

    virtual ucharArr loadUCharArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;
    virtual floatArr loadFloatArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;
    virtual doubleArr loadDoubleArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;

    virtual void saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<float>& data) const;
    virtual void saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<double>& data) const;
    virtual void saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned char>& data) const;

protected:
    FeatureBase* m_featureBase= static_cast<FeatureBase*>(this);

};

} // namespace lvr2

#include "ArrayIO.tcc"

#endif

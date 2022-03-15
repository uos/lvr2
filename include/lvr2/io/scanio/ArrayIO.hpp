#ifndef LVR2_IO_scanio_ARRAYIO_HPP
#define LVR2_IO_scanio_ARRAYIO_HPP

#include "lvr2/io/scanio/FeatureBase.hpp"
#include <boost/shared_array.hpp>

namespace lvr2 
{

namespace scanio
{

template<typename FeatureBase>
class ArrayIO {
public:

    ucharArr loadUCharArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    floatArr loadFloatArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;
    doubleArr loadDoubleArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;
    intArr loadIntArray(const std::string& group, const std::string& container, std::vector<size_t> &dims) const;

    void saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<float>& data) const;
    void saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<double>& data) const;
    void saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned char>& data) const;
    void saveIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<int>& data) const;

protected:
    FeatureBase* m_featureBase= static_cast<FeatureBase*>(this);

};

} // namespace scanio

} // namespace lvr2

#include "ArrayIO.tcc"

#endif // LVR2_IO_scanio_ARRAYIO_HPP

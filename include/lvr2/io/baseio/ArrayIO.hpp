#ifndef ARRAYIO
#define ARRAYIO

#include "lvr2/io/baseio/BaseIO.hpp"
#include <boost/shared_array.hpp>

namespace lvr2 
{

namespace baseio
{

template<typename BaseIO>
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
    BaseIO* m_BaseIO= static_cast<BaseIO*>(this);

};

} // namespace baseio

} // namespace lvr2

#include "ArrayIO.tcc"

#endif // ARRAYIO

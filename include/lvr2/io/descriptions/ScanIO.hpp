#pragma once

#ifndef LVR2_IO_HDF5_SCANIO_HPP
#define LVR2_IO_HDF5_SCANIO_HPP

#include "ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <sstream>
#include <yaml-cpp/yaml.h>
namespace lvr2
{


template <typename FeatureBase>
class ScanIO
{
  public:
   // void save(const std::string& group, const std::string& container, const std::string& metaFile, const ScanPtr& scan);
    void saveScan(const size_t& scanPosNo, const size_t& scanNo, const ScanPtr& buffer);
  
    ScanPtr loadScan(const size_t& scanPosNo, const size_t& scanNo);
    //ScanPtr load(const std::string& group, const std::string& container);
    
  protected:
    bool isScan(const std::string& group);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    ArrayIO<FeatureBase>* m_arrayIO = static_cast<ArrayIO<FeatureBase>*>(m_featureBase);
    MatrixIO<FeatureBase>* m_matrixIO = static_cast<MatrixIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanIO";
    static constexpr const char* OBJID = "Scan";
};

/**
 *
 * @brief FeatureConstruct Specialization for ScanIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<ScanIO, FeatureBase >
{

    // DEPS
    using dep1 = typename FeatureConstruct<ArrayIO, FeatureBase >::type;
    using dep2 = typename FeatureConstruct<MatrixIO, FeatureBase >::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<ScanIO>::type;
};

} // namespace lvr2

#include "ScanIO.tcc"

#endif // LVR2_IO_HDF5_SCANIO_HPP

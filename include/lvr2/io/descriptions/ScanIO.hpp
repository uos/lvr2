#pragma once

#ifndef LVR2_IO_HDF5_SCANIO_HPP
#define LVR2_IO_HDF5_SCANIO_HPP

#include "lvr2/io/descriptions/ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "WaveformIO.hpp"
#include "PointCloudIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

#include <sstream>
#include <yaml-cpp/yaml.h>
namespace lvr2
{

template <typename FeatureBase>
class ScanIO
{
  public:
  
    void save(
            const size_t& scanPosNo, 
            const size_t& scanNo, 
            ScanPtr buffer
        ) const;
    
    void saveScan(
            const size_t& scanPosNo, 
            const size_t& scanNo, 
            ScanPtr buffer
        ) const;
  
    ScanPtr loadScan(
            const size_t& scanPosNo, 
            const size_t& scanNo
        ) const;

    ScanPtr loadScan(
            const size_t& scanPosNo, 
            const size_t& scanNo, 
            ReductionAlgorithmPtr reduction
        ) const;

  protected:
    bool isScan(const std::string& group);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    PointCloudIO<FeatureBase>* m_pclIO = static_cast<PointCloudIO<FeatureBase>*>(m_featureBase);
    FullWaveformIO<FeatureBase>* m_fullWaveformIO =
        static_cast<FullWaveformIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanIO";
    static constexpr const char* OBJID = "Scan";
};

/**
 *
 * @brief FeatureConstruct Specialization for ScanIO
 * - Constructs dependencies (PointCloudIO, FullWaveformIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<ScanIO, FeatureBase >
{
    // DEPS
    using dep1 = typename FeatureConstruct<PointCloudIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<FullWaveformIO, FeatureBase >::type;
    
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<ScanIO>::type;
};

} // namespace lvr2

#include "ScanIO.tcc"

#endif // LVR2_IO_HDF5_SCANIO_HPP

#pragma once

#ifndef LVR2_IO_HDF5_SCANCAMERAIO_HPP
#define LVR2_IO_HDF5_SCANCAMERAIO_HPP

#include "ScanImageIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class ScanCameraIO
{
  public:
    void saveScanCamera(const size_t& scanPosNo, const size_t& scanCamNo, ScanCameraPtr& camera);
    //void save(const std::string& group, const std::string& container, ScanCameraPtr& buffer);
    //ScanCameraPtr load(const std::string& group, const std::string& constainer);
    ScanCameraPtr loadScanCamera(const size_t& scanPosNo, const size_t& scanCamNo);

  protected:
    bool isScanCamera(const std::string& group);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    ScanImageIO<FeatureBase>* m_scanImageIO = static_cast<ScanImageIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanCameraIO";
    static constexpr const char* OBJID = "ScanCamera";
};

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanCameraIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<ScanCameraIO, FeatureBase>
{

    // DEPS
    using deps = typename FeatureConstruct<ScanImageIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<ScanCameraIO>::type;
};

} // namespace lvr2

#include "ScanCameraIO.tcc"

#endif // LVR2_IO_HDF5_SCANCAMERAIO_HPP

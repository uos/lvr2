#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_CAMERAIO_HPP
#define LVR2_IO_DESCRIPTIONS_CAMERAIO_HPP

#include "CameraImageIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class CameraIO
{
  public:

    void save(
        const size_t& scanPosNo,
        const size_t& scanCamNo,
        CameraPtr cameraPtr) const;

    CameraPtr load(
        const size_t& scanPosNo,
        const size_t& scanCamNo) const;

    void saveCamera(
        const size_t& scanPosNo, 
        const size_t& scanCamNo, 
        CameraPtr cameraPtr) const;

    CameraPtr loadCamera(
        const size_t& scanPosNo, 
        const size_t& scanCamNo) const;

  protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    CameraImageIO<FeatureBase>* m_cameraImageIO = static_cast<CameraImageIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "CameraIO";
    static constexpr const char* OBJID = "Camera";
};

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanCameraIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<CameraIO, FeatureBase>
{

    // DEPS
    using deps = typename FeatureConstruct<CameraImageIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<CameraIO>::type;
};

} // namespace lvr2

#include "CameraIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CAMERAIO_HPP

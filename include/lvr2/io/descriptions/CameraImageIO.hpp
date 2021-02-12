#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP
#define LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP

#include "ImageIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class CameraImageIO
{
public:

    void save(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr,
        CameraImagePtr imgPtr
    ) const;

    CameraImagePtr load(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr
    ) const;

    void saveCameraImage(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr,
        CameraImagePtr imgPtr
    ) const;

    CameraImagePtr loadCameraImage(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr) const;
    
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    ImageIO<FeatureBase>* m_imageIO = static_cast<ImageIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanImageIO";
    static constexpr const char* OBJID = "ScanImage";
};

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanImageIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<CameraImageIO, FeatureBase>
{
    // DEPS
    using deps = typename FeatureConstruct<ImageIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<CameraImageIO>::type;
};

} // namespace lvr2

#include "CameraImageIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP

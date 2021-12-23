#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP
#define LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP

#include "MetaIO.hpp"
#include "ImageIO.hpp"
#include "FeatureBase.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/CameraImage.hpp"

namespace lvr2
{

template <typename FeatureBase>
class CameraImageIO
{
public:

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo,
        CameraImagePtr imgPtr
    ) const;

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos,
        CameraImagePtr imgPtr
    ) const;

    CameraImagePtr load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo
    ) const;

    CameraImagePtr load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos
    ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo
    ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos
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
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
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
    using dep1 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<ImageIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<CameraImageIO>::type;
};

} // namespace lvr2

#include "CameraImageIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CAMERAIMAGEIO_HPP
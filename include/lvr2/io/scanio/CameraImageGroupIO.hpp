#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_CAMERAIMAGEGROUPIO_HPP
#define LVR2_IO_DESCRIPTIONS_CAMERAIMAGEGROUPIO_HPP

#include "MetaIO.hpp"
#include "CameraImageIO.hpp"
#include "FeatureBase.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/CameraImage.hpp"

namespace lvr2
{

template <typename FeatureBase>
class CameraImageGroupIO
{
public:

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo,
        CameraImageGroupPtr imgPtr
    ) const;

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos,
        CameraImageGroupPtr imgPtr
    ) const;

    CameraImageGroupPtr load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo
    ) const;

    CameraImageGroupPtr load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos
    ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& imgNo) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const std::vector<size_t>& imgNos
    ) const;

    void saveCameraImage(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr,
        CameraImageGroupPtr imgPtr
    ) const;

    CameraImageGroupPtr loadCameraImage(
        const size_t& scanPosNr,
        const size_t& camNr,
        const size_t& imgNr) const;
    
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    CameraImageIO<FeatureBase>* m_cameraImageIO = static_cast<CameraImageIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "CameraImageGroupIO";
    static constexpr const char* OBJID = "CameraImageGroup";
};

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanImageIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<CameraImageGroupIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<CameraImageIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<CameraImageGroupIO>::type;
};

} // namespace lvr2

#include "CameraImageGroupIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CAMERAIMAGEGROUPIO_HPP

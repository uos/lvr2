#pragma once

#ifndef CAMERAIO
#define CAMERAIO

#include "MetaIO.hpp"
#include "CameraImageIO.hpp"
#include "CameraImageGroupIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/Camera.hpp"


namespace lvr2
{

namespace scanio
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

    boost::optional<YAML::Node> loadMeta(
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
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    CameraImageIO<FeatureBase>* m_cameraImageIO = static_cast<CameraImageIO<FeatureBase>*>(m_featureBase);
    CameraImageGroupIO<FeatureBase>* m_cameraImageGroupIO = static_cast<CameraImageGroupIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "CameraIO";
    static constexpr const char* OBJID = "Camera";
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanCameraIO
 * - Constructs dependencies (CameraImageIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::CameraIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::CameraImageIO, FeatureBase>::type;
    using dep3 = typename FeatureConstruct<lvr2::scanio::CameraImageGroupIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::CameraIO>::type;
};

} // namespace lvr2

#include "CameraIO.tcc"

#endif // CAMERAIO

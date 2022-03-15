#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRALCAMERAIO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRALCAMERAIO_HPP

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
#include "MetaIO.hpp"
#include "HyperspectralPanoramaIO.hpp"


namespace lvr2
{

namespace scanio
{

template <typename FeatureBase>
class HyperspectralCameraIO
{
public:
    void save(
        const size_t& scanPosNo, 
        const size_t& hCamNo, 
        HyperspectralCameraPtr hcam) const;

    HyperspectralCameraPtr load(
        const size_t& scanPosNo,
        const size_t& hCamNo) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& hCamNo) const;

protected:
    FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    HyperspectralPanoramaIO<FeatureBase>* m_hyperspectralPanoramaIO = static_cast<HyperspectralPanoramaIO<FeatureBase>*>(m_featureBase);

    // dependencies
    static constexpr const char *ID = "HyperspectralCameraIO";
    static constexpr const char *OBJID = "HyperspectralCamera";
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for HyperspectralCameraIO
 * - Constructs dependencies (HyperspectralPanoramaIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::HyperspectralCameraIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::HyperspectralPanoramaIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;
    
    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::HyperspectralCameraIO>::type;
};

} // namespace lvr2

#include "HyperspectralCameraIO.tcc"

#endif // LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP

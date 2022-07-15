#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRALCAMERAIO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRALCAMERAIO_HPP

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
#include "lvr2/io/scanio/HyperspectralPanoramaIO.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
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
    BaseIO *m_baseIO = static_cast<BaseIO*>(this);

    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    HyperspectralPanoramaIO<BaseIO>* m_hyperspectralPanoramaIO = static_cast<HyperspectralPanoramaIO<BaseIO>*>(m_baseIO);

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
template <typename T>
struct FeatureConstruct<lvr2::scanio::HyperspectralCameraIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::HyperspectralPanoramaIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;
    
    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::HyperspectralCameraIO>::type;
};

} // namespace lvr2

#include "HyperspectralCameraIO.tcc"

#endif // LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP

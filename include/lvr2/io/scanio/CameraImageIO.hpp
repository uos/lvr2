#pragma once

#ifndef CAMERAIMAGEIO
#define CAMERAIMAGEIO

#include "ImageIO.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/scanio/yaml/CameraImage.hpp"
#include "lvr2/types/ScanTypes.hpp"

using lvr2::baseio::FeatureConstruct;
using lvr2::baseio::FeatureBuild;
using lvr2::baseio::MetaIO;

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
class CameraImageIO
{
public:

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo,
        const size_t& imgNo,
        CameraImagePtr imgPtr
    ) const;

    // void save(
    //     const size_t& scanPosNo,
    //     const size_t& camNo,
    //     const size_t& groupNo,
    //     const std::vector<size_t>& imgNos,
    //     const std::vector<CameraImagePtr>& imgPtr
    // ) const;

    void saveCameraImage(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t& groupNo,
        const size_t &imgNo,
        CameraImagePtr imgPtr) const;

    CameraImagePtr load(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t& groupNo,
        const size_t &imgNo) const;

    std::vector<CameraImagePtr> load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo,
        const std::vector<size_t>& imgNos
    ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo,
        const std::vector<size_t>& imgNos
    ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t& groupNo,
        const size_t &imgNo
    ) const;

    CameraImagePtr loadCameraImage(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo,
        const size_t& imgNo) const;
    
protected:

    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    ImageIO<BaseIO>* m_imageIO = static_cast<ImageIO<BaseIO>*>(m_baseIO);

    static constexpr const char* ID = "ScanImageIO";
    static constexpr const char* OBJID = "ScanImage";
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanImageIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename T>
struct FeatureConstruct<lvr2::scanio::CameraImageIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::ImageIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::CameraImageIO>::type;
};

} // namespace lvr2

#include "CameraImageIO.tcc"

#endif // CAMERAIMAGEIO

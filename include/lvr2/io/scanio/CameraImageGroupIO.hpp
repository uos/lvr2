#pragma once

#ifndef CAMERAIMAGEGROUPIO
#define CAMERAIMAGEGROUPIO

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/scanio/yaml/CameraImage.hpp"
#include "lvr2/io/scanio/CameraImageIO.hpp"

using lvr2::baseio::FeatureConstruct;

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
class CameraImageGroupIO
{
public:

    void save(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo,
        CameraImageGroupPtr groupPtr) const;

    void save(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t& groupNo,
        const std::vector<size_t> &imgNos,
        CameraImageGroupPtr imgPtr) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo) const;

    CameraImageGroupPtr load(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& groupNo) const;
    
protected:
    BaseIO* m_baseIO = static_cast< BaseIO*>(this);

    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    CameraImageIO<BaseIO>* m_cameraImageIO = static_cast<CameraImageIO<BaseIO>*>(m_baseIO);

    static constexpr const char* ID = "CameraImageGroupIO";
    static constexpr const char* OBJID = "CameraImageGroup";
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
struct FeatureConstruct<lvr2::scanio::CameraImageGroupIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::CameraImageIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::CameraImageGroupIO>::type;
};

} // namespace lvr2

#include "CameraImageGroupIO.tcc"

#endif // CAMERAIMAGEGROUPIO

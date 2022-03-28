#pragma once 

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/CameraImageIO.hpp"
#include "lvr2/io/scanio/CameraImageIO.hpp"
#include "lvr2/io/scanio/CameraImageGroupIO.hpp"
#include "lvr2/io/scanio/yaml/Camera.hpp"

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
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
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);

    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    CameraImageIO<BaseIO>* m_cameraImageIO = static_cast<CameraImageIO<BaseIO>*>(m_baseIO);
    CameraImageGroupIO<BaseIO>* m_cameraImageGroupIO = static_cast<CameraImageGroupIO<BaseIO>*>(m_baseIO);

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
template <typename T>
struct FeatureConstruct<lvr2::scanio::CameraIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::CameraImageIO, T>::type;
    using dep3 = typename FeatureConstruct<lvr2::scanio::CameraImageGroupIO, T>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::CameraIO>::type;
};

} // namespace lvr2

#include "CameraIO.tcc"



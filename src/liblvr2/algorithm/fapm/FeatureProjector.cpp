#include "lvr2/algorithm/fapm/FeatureProjector.hpp"

namespace lvr2 {

FeatureProjector::FeatureProjector(
        RaycasterBasePtr raycaster,
        ScanCamera scan_cam,
        cv::Ptr<cv::Feature2D> feature
    )
:FeatureProjector(raycaster, feature, feature)
{
    
}

FeatureProjector::FeatureProjector(
    lvr2::RaycasterBasePtr raycaster,
    cv::Ptr<cv::Feature2D> feature_detector,
    cv::Ptr<cv::Feature2D> feature_descriptor)
: m_raycaster(raycaster)
, m_feature_detector(feature_detector)
, m_feature_descriptor(feature_descriptor)
{
    std::cout << "init Feature Projector" << std::endl; 
}

void FeatureProjector::projectPixels(
    const std::vector<cv::Point2f>& pixels,
    const Extrinsicsd& T,
    const Intrinsicsd& M,
    std::vector<uint8_t>& hits,
    std::vector<lvr2::Vector2f>& intersections)
{

}

} // namespace lvr2
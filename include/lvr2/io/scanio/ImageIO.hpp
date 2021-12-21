#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_IMAGEIO_HPP
#define LVR2_IO_DESCRIPTIONS_IMAGEIO_HPP

#include <boost/optional.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/traits.hpp>

namespace lvr2 {

template<typename FeatureBase>
class ImageIO {
public:

    void save(      
        std::string groupName, 
        std::string datasetName, 
        const cv::Mat& img
    ) const;

    boost::optional<cv::Mat> load(
        std::string groupName,
        std::string datasetName
    ) const;

    void saveImage( 
        std::string groupName,
        std::string datasetName,
        const cv::Mat& img
    ) const;

    boost::optional<cv::Mat> loadImage(
        std::string groupName,
        std::string datasetName
    ) const;

protected:

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};

} // namespace lvr2

#include "ImageIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_IMAGEIO_HPP
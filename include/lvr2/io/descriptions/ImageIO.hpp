#pragma once

#ifndef LVR2_IO_HDF5_IMAGEIO_HPP
#define LVR2_IO_HDF5_IMAGEIO_HPP

#include <boost/optional.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/traits.hpp>

namespace lvr2 {

template<typename FeatureBase>
class ImageIO {
public:

    void saveImage(std::string groupName,
        std::string datasetName,
        const cv::Mat& img
    );

    boost::optional<cv::Mat> loadImage(std::string groupName,
        std::string datasetName);

protected:

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};

} // namespace lvr2

#include "ImageIO.tcc"

#endif // LVR2_IO_HDF5_IMAGEIO_HPP
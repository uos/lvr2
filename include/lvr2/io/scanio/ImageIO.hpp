#pragma once

#ifndef IMAGEIO
#define IMAGEIO

#include <boost/optional.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/traits.hpp>

namespace lvr2 {

namespace scanio
{

template<typename BaseIO>
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

    BaseIO* m_baseIO = static_cast<BaseIO*>(this);

};

} // namespace scanio

} // namespace lvr2

#include "ImageIO.tcc"

#endif // IMAGEIO

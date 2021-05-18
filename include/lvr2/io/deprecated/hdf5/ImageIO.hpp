#pragma once

#ifndef LVR2_IO_HDF5_IMAGEIO_HPP
#define LVR2_IO_HDF5_IMAGEIO_HPP

#include <boost/optional.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/traits.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class ImageIO {
public:

    void save(std::string groupName,
        std::string datasetName,
        const cv::Mat& img
    );

    void save(HighFive::Group& group,
        std::string datasetName,
        const cv::Mat& img
    );

    boost::optional<cv::Mat> load(HighFive::Group& group,
        std::string datasetName);

    boost::optional<cv::Mat> load(std::string groupName,
        std::string datasetName);

    boost::optional<cv::Mat> loadImage(std::string groupName,
        std::string datasetName);

protected:

    template<typename T>
    cv::Mat createMat(std::vector<size_t>& dims);

    Derived* m_file_access = static_cast<Derived*>(this);

};

} // namespace hdf5features

} // namespace lvr2

#include "ImageIO.tcc"

#endif // LVR2_IO_HDF5_IMAGEIO_HPP
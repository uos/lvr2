
namespace lvr2 {

namespace scanio
{

template<typename BaseIO>
void ImageIO<BaseIO>::save(std::string groupName,
    std::string datasetName,
    const cv::Mat& img) const
{
    m_baseIO->m_kernel->saveImage(groupName, datasetName, img);
}

template<typename BaseIO>
boost::optional<cv::Mat> ImageIO<BaseIO>::load(std::string groupName,
    std::string datasetName) const
{
    return m_baseIO->m_kernel->loadImage(groupName, datasetName);
}

template<typename BaseIO>
void ImageIO<BaseIO>::saveImage(std::string groupName,
    std::string datasetName,
    const cv::Mat& img) const
{
    save(groupName, datasetName, img);
}

template<typename BaseIO>
boost::optional<cv::Mat> ImageIO<BaseIO>::loadImage(std::string groupName,
    std::string datasetName) const
{
    return load(groupName, datasetName);
}

} // namespace scanio

} // namespace lvr2
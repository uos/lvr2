
namespace lvr2 {

namespace scanio
{

template<typename FeatureBase>
void ImageIO<FeatureBase>::save(std::string groupName,
    std::string datasetName,
    const cv::Mat& img) const
{
    m_featureBase->m_kernel->saveImage(groupName, datasetName, img);
}

template<typename FeatureBase>
boost::optional<cv::Mat> ImageIO<FeatureBase>::load(std::string groupName,
    std::string datasetName) const
{
    return m_featureBase->m_kernel->loadImage(groupName, datasetName);
}

template<typename FeatureBase>
void ImageIO<FeatureBase>::saveImage(std::string groupName,
    std::string datasetName,
    const cv::Mat& img) const
{
    save(groupName, datasetName, img);
}

template<typename FeatureBase>
boost::optional<cv::Mat> ImageIO<FeatureBase>::loadImage(std::string groupName,
    std::string datasetName) const
{
    return load(groupName, datasetName);
}

} // namespace scanio

} // namespace lvr2
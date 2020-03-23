
namespace lvr2 {

template<typename FeatureBase>
void ImageIO<FeatureBase>::save(std::string groupName,
    std::string datasetName,
    const cv::Mat& img)
{
    m_featureBase->m_kernel->saveImage(groupName, datasetName, img);
}


template<typename FeatureBase>
boost::optional<cv::Mat> ImageIO<FeatureBase>::load(std::string groupName,
    std::string datasetName)
{
    return m_featureBase->m_kernel->loadImage(groupName, datasetName);
}

} // namespace lvr2
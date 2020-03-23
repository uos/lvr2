namespace lvr2
{

template <typename FeatureBase>
void ScanCameraIO<FeatureBase>::save(
    const std::string& group, 
    const std::string& container, ScanCameraPtr& buffer)
{
    // TODO
}

template <typename FeatureBase>
ScanCameraPtr ScanCameraIO<FeatureBase>::load(
    const std::string& group, 
    const std::string& container)
{
    ScanCameraPtr ret;

    // TODO

    return ret;
}



template <typename FeatureBase>
bool ScanCameraIO<FeatureBase>::isScanCamera(const std::string& group)
{
    return true;
}

} // namespace lvr2

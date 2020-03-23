namespace lvr2
{

template <typename FeatureBase>
void ScanImageIO<FeatureBase>::save(
    const std::string& group, 
    const std::string& container, 
    const ScanImagePtr& buffer)
{
    // TODO
}



template <typename FeatureBase>
ScanImagePtr ScanImageIO<FeatureBase>::load(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr)
{
    ScanImagePtr ret;

    // TODO

    return ret;
}

template <typename FeatureBase>
ScanImagePtr ScanImageIO<FeatureBase>::load(
    const std::string& group, 
    const std::string& container)
{
    ScanImagePtr ret;

    // check wether the given group is type ScanProjectIO

    // TODO

    return ret;
}

template <typename FeatureBase>
bool ScanImageIO<FeatureBase>::isScanImage(HighFive::Group& group)
{
    return true;
}

} // namespace lvr2

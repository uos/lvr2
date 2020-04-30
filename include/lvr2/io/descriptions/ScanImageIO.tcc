namespace lvr2
{


template <typename FeatureBase>
ScanImagePtr ScanImageIO<FeatureBase>::loadScanImage(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr)
{
    ScanImagePtr ret;

    // TODO

    return ret;
}

template <typename FeatureBase>
void  ScanImageIO<FeatureBase>::saveScanImage(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr, 
    ScanImagePtr& buffer)
{
    // TODO
}

// template <typename FeatureBase>
// ScanImagePtr ScanImageIO<FeatureBase>::loadScanImage(
//     const std::string& group, 
//     const std::string& container)
// {
//     ScanImagePtr ret;

//     // check wether the given group is type ScanProjectIO

//     // TODO

//     return ret;
// }



} // namespace lvr2

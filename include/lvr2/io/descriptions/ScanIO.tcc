namespace lvr2
{

template <typename FeatureBase>
void ScanIO<FeatureBase>::save(const size_t& scanPosNo, const size_t& scanNo, const ScanPtr& scanPtr)
{
    // Setup defaults: no group and scan number into .ply file
    std::string groupName;


    std::stringstream sstr;
    sstr << "scan" << std::setfill('0') << std::setw(8) << scanNo << ".ply" << std::endl;
    std::string scanName = sstr.str();

    // Get group and dataset names according to 
    // data fomat description and override defaults if 
    // when possible
    Description d = m_featureBase->m_description->scan(scanPosNo, scanNo)

    if(d.groupName)
    {
        groupName = d.groupName;
    }

    if(d.dataSetName)
    {
        scanName = d.dataSetName;
    }

    // TODO: META INFORMATION!!!!

    // Forwars saving
    save(groupName, scanName, scanPtr);
}

template <typename FeatureBase>
void ScanIO<FeatureBase>::save(const std::string& group, const std::string& name, const ScanPtr& scanPtr)
{
 
}


template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::load(const size_t& scanPos, const size_t& scanNo)
{
    ScanPtr ret;
    return ret;
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::load(const std::string& group, const std::string& name)
{
    ScanPtr ret;
    return ret;
}


template <typename FeatureBase>
bool ScanIO<FeatureBase>::isScan(HighFive::Group& group)
{
    return true;
}

} // namespace lvr2

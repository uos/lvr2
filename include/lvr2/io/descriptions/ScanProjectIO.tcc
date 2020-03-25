namespace lvr2
{

template <typename FeatureBase>
void ScanProjectIO<FeatureBase>::save(const ScanProjectPtr& scanProjectPtr)
{
    Description d = m_featureBase->m_description->scanproject();

    // Check for meta data and save
    if(d.metaData)
    {
        saveMetaYAML(*d.metaName);
    }
    else
    {
        if(d.metaName)
        {
            // Create default meta and save
            YAML::node node = *scanProjectPtr;
            saveMetaYAML(*d.metaName, node);
        }
       
    }   
    
    // Iterate over all positions and save
    for (size_t i = 0; i < scanProjectPtr->positions.size())
    {
        m_scanPositionIO->save(i,  scanProjectPtr->positions[i]);
    }
}

template <typename FeatureBase>
ScanProjectPtr ScanProjectIO<FeatureBase>::load()
{
    ScanProjectPtr ret(new ScanProject());

    // Load description and meta data for scan project
    Description d = m_featureBase->m_description->scanproject();
    if(d.metaData)
    {
        ret = *(d.metaData);
    }

    // Get all sub scans
    size_t scanPos = 0;
    do
    {
        // Get description for next scan
        Description scanDescr = m_featureBase->m_description->scanposition(scanPosNo);

        // Check if it exists. If not, exit.
        if(m_featureBase->m_kernel->exists(scanDescr.groupName)
        {
            std::cout << timestamp 
                      << "ScanPositionIO: Loading scanposition " 
                      << scanPosNo << std::endl;
                  
            ScanPositionPtr scanPos = m_scanPositionIO->load(scanPosNo);
            ret->positions->push_back(scanPos);
        }
        else
        {
            break;
        }
        ++scanPos;
    } 
    while (true);

    return ret;
}


} // namespace lvr2

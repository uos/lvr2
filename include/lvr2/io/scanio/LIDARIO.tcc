namespace lvr2 {

template <typename FeatureBase>
void LIDARIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    LIDARPtr lidar) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->lidar(scanPosNo, lidarNo);

    // std::cout << "LIDARIO - save data " << std::endl;
    // Save data
    for(size_t scanNo = 0; scanNo < lidar->scans.size(); scanNo++)
    {
        m_scanIO->save(scanPosNo, lidarNo, scanNo, lidar->scans[scanNo]);
    }

    // std::cout << "LIDARIO - save meta " << std::endl;
    // Save meta
    if(d.meta)
    {
        YAML::Node node;
        node = *lidar;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename FeatureBase>
boost::optional<YAML::Node> LIDARIO<FeatureBase>::loadMeta(
        const size_t& scanPosNo,
        const size_t& lidarNo) const
{
    Description d = m_featureBase->m_description->lidar(scanPosNo, lidarNo);
    return m_metaIO->load(d);
}

template <typename FeatureBase>
LIDARPtr LIDARIO<FeatureBase>::load(
    const size_t& scanPosNo,
    const size_t& lidarNo) const
{
    LIDARPtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->lidar(scanPosNo, lidarNo);

    if(!d.dataRoot)
    {
        return ret;
    }

    // check if group exists
    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    // std::cout << "[LIDARIO - load] Description:" << std::endl;
    // std::cout << d << std::endl;

    ///////////////////////
    //////  META
    ///

    if(d.meta)
    {
        YAML::Node meta;
        if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        ret = std::make_shared<LIDAR>(meta.as<LIDAR>());
    } else {
        // no meta name specified but scan position is there: 
        ret.reset(new LIDAR);
    }

    ///////////////////////
    //////  DATA
    ///

    size_t scanNo = 0;
    while(true)
    {
        // std::cout << "[LIDARIO - load] Load Scan " << scanNo << std::endl;
        ScanPtr scan = m_scanIO->load(scanPosNo, lidarNo, scanNo);
        if(scan)
        {
            ret->scans.push_back(scan);
        } else {
            break;
        }
        scanNo++;
    }
    return ret;
}


} // namespace lvr2
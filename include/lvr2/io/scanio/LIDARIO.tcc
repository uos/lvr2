namespace lvr2 {

namespace scanio
{

template <typename BaseIO>
void LIDARIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    LIDARPtr lidar) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->lidar(scanPosNo, lidarNo);

    // std::cout << "LIDARIO - save data " << std::endl;
    // Save data
    for(size_t scanNo = 0; scanNo < lidar->scans.size(); scanNo++)
    {
        // std::cout << "[LIDARIO - save] Save Scan " << scanNo << std::endl;
        m_scanIO->save(scanPosNo, lidarNo, scanNo, lidar->scans[scanNo]);
    }

    // std::cout << "LIDARIO - save meta " << std::endl;
    // Save meta
    if(d.meta)
    {
        YAML::Node node;
        node = *lidar;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename BaseIO>
boost::optional<YAML::Node> LIDARIO<BaseIO>::loadMeta(
        const size_t& scanPosNo,
        const size_t& lidarNo) const
{
    Description d = m_baseIO->m_description->lidar(scanPosNo, lidarNo);
    return m_metaIO->load(d);
}

template <typename BaseIO>
LIDARPtr LIDARIO<BaseIO>::load(
    const size_t& scanPosNo,
    const size_t& lidarNo) const
{
    LIDARPtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->lidar(scanPosNo, lidarNo);

    if(!d.dataRoot)
    {
        return ret;
    }

    // check if group exists
    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    std::cout << "[LIDARIO - load] Description:" << std::endl;
    std::cout << d << std::endl;

    ///////////////////////
    //////  META
    ///

    if(d.meta)
    {
        YAML::Node meta;
        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        // std::cout << "[LIDARIO - load] Load Meta " << std::endl;
        ret = std::make_shared<LIDAR>(meta.as<LIDAR>());
    } else {
        // no meta name specified but scan position is there: 
        ret = std::make_shared<LIDAR>();
    }

    ///////////////////////
    //////  DATA
    ///

    size_t scanNo = 0;
    while(true)
    {
        std::cout << "[LIDARIO - load] Load Scan " << scanNo << std::endl;
        ScanPtr scan = m_scanIO->load(scanPosNo, lidarNo, scanNo);
        if(scan)
        {
            ret->scans.push_back(scan);
        } else {
            break;
        }

        std::cout << "[LIDARIO - load] Loaded Scan " << scanNo << std::endl;
        scanNo++;
    }
    return ret;
}

} // namespace scanio

} // namespace lvr2
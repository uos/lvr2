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

    //lvr2::logout::get() << "LIDARIO - save data " << lvr2::endl;
    // Save data
    for(size_t scanNo = 0; scanNo < lidar->scans.size(); scanNo++)
    {
        // lvr2::logout::get() << "[LIDARIO - save] Save Scan " << scanNo << lvr2::endl;
        m_scanIO->save(scanPosNo, lidarNo, scanNo, lidar->scans[scanNo]);
    }

    // lvr2::logout::get() << "LIDARIO - save meta " << lvr2::endl;
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
    // lvr2::logout::get() << "LIDARIO - save data " << lvr2::endl;

    LIDARPtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->lidar(scanPosNo, lidarNo);

    if (!d.dataRoot)
    {

        return ret;
    }
    // check if group exists
    if (!m_baseIO->m_kernel->exists(*d.dataRoot))
    {

        return ret;
    }

    // lvr2::logout::get() << "[LIDARIO - load] Description:" << lvr2::endl;
    // lvr2::logout::get() << d << lvr2::endl;

    ///////////////////////
    //////  META
    ///

    if (d.meta)
    {

        YAML::Node meta;
        if (!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        // lvr2::logout::get() << "[LIDARIO - load] Load Meta " << lvr2::endl;
        try
        {
            ret = std::make_shared<LIDAR>(meta.as<LIDAR>());
        }
        catch (const YAML::TypedBadConversion<LIDAR> &ex)
        {
            lvr2::logout::get() << lvr2::error << "[LIDARIO - load] LIDAR (" << scanPosNo << ", " << lidarNo << ") : Could not decode YAML as LIDAR." << lvr2::endl;
            throw ex;
        }
    }
    else
    {
        // no meta name specified but scan position is there:
        ret = std::make_shared<LIDAR>();
    }

    ///////////////////////
    //////  DATA
    ///

    size_t scanNo = 0;
    while (true)
    {
        // lvr2::logout::get() << "[LIDARIO - load] Load Scan " << scanNo << lvr2::endl;
        ScanPtr scan = m_scanIO->load(scanPosNo, lidarNo, scanNo);
        if (scan)
        {
            ret->scans.push_back(scan);
        }
        else
        {
            break;
        }

        // lvr2::logout::get() << "[LIDARIO - load] Loaded Scan " << scanNo << lvr2::endl;
        scanNo++;
    }
    return ret;
}

} // namespace scanio

} // namespace lvr2
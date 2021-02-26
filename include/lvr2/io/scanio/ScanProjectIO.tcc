namespace lvr2
{

template<typename FeatureBase>
void ScanProjectIO<FeatureBase>::save(
    ScanProjectPtr scanProject) const
{
    Description d = m_featureBase->m_description->scanProject();


    // std::cout << "[ScanProjectIO - save]: Description" << std::endl;
    // std::cout << d << std::endl;

    // if(!d.dataRoot)
    // {
    //     std::cout << timestamp << "[ScanProjectIO] Description does not contain a data root for the ScanProject" << std::endl;
    //     d.dataRoot = "";
    // }

    // std::cout << "[ScanProjectIO] Save Scan Project "<< std::endl;
    // Iterate over all positions and save
    for (size_t i = 0; i < scanProject->positions.size(); i++)
    {
        // std::cout << "[ScanProjectIO] Save Pos" << i << std::endl;
        m_scanPositionIO->saveScanPosition(i, scanProject->positions[i]);
    }

    // Default scan project yaml
    if(d.meta)
    {
        YAML::Node node;
        node = *scanProject;
        // std::cout << "[ScanProjectIO] saveMetaYAML, Group: "
        //             << *d.groupName << ", metaName: " << *d.metaName << std::endl;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename FeatureBase>
boost::optional<YAML::Node> ScanProjectIO<FeatureBase>::loadMeta() const
{
    Description d = m_featureBase->m_description->scanProject();
    return m_metaIO->load(d);
}

template <typename FeatureBase>
ScanProjectPtr ScanProjectIO<FeatureBase>::load() const
{
    ScanProjectPtr ret;

    // Load description and meta data for scan project
    Description d = m_featureBase->m_description->scanProject();

    // std::cout << "[HDF5IO - ScanProjectIO - load]: Description" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    if(*d.dataRoot != "" && !m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        std::cout << "[ScanProjectIO] Warning: '" << *d.dataRoot << "' does not exist." << std::endl; 
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);
        ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
    } 
    else 
    {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        // std::cout << timestamp << "[ScanProjectIO] Could not load meta information. No meta name specified." << std::endl;
        ret.reset(new ScanProject);
    }


    // Get all sub scans
    size_t scanPosNo = 0;
    while(true)
    {  
        // Get description for next scan
        ScanPositionPtr scanPos = m_scanPositionIO->loadScanPosition(scanPosNo);
        if(!scanPos)
        {
            break;
        }

        // std::cout << "[ScanProjectIO - load] loaded ScanPosition "  << scanPosNo << std::endl;
        ret->positions.push_back(scanPos);
        scanPosNo++;
    }

    return ret;
}

template <typename FeatureBase>
void ScanProjectIO<FeatureBase>::saveScanProject(
    ScanProjectPtr scanProject) const
{
    save(scanProject);
}

template <typename FeatureBase>
ScanProjectPtr ScanProjectIO<FeatureBase>::loadScanProject() const
{
    return load();
}

template <typename FeatureBase>
ScanProjectPtr ScanProjectIO<FeatureBase>::loadScanProject(ReductionAlgorithmPtr reduction) const
{
    ScanProjectPtr ret;

    // Load description and meta data for scan project
    Description d = m_featureBase->m_description->scanProject();

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    if(*d.dataRoot != "" && !m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        std::cout << "[ScanProjectIO] Warning: '" << *d.dataRoot << "' does not exist." << std::endl; 
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);
        ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
    } 
    else 
    {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        // std::cout << timestamp << "[ScanProjectIO] Could not load meta information. No meta name specified." << std::endl;
        ret.reset(new ScanProject);
    }


    // Get all sub scans
    size_t scanPosNo = 0;
    while(true)
    {
        // Get description for next scan
        ScanPositionPtr scanPos = m_scanPositionIO->loadScanPosition(scanPosNo, reduction);
        if(!scanPos)
        {
            break;
        }
        ret->positions.push_back(scanPos);
        scanPosNo++;
    }

    return ret;
}

} // namespace lvr2

namespace lvr2
{
namespace scanio
{

template<typename BaseIO>
void ScanProjectIO<BaseIO>::save(
    ScanProjectPtr scanProject) const
{
    Description d = m_baseIO->m_description->scanProject();


//     std::cout << "[ScanProjectIO - save]: Description" << std::endl;
//     std::cout << d << std::endl;
//
//     if(!d.dataRoot)
//     {
//         std::cout << timestamp << "[ScanProjectIO] Description does not contain a data root for the ScanProject" << std::endl;
//         d.dataRoot = "";
//     }
//
//     std::cout << "[ScanProjectIO] Save Scan Project "<< std::endl;
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
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename BaseIO>
boost::optional<YAML::Node> ScanProjectIO<BaseIO>::loadMeta() const
{
    Description d = m_baseIO->m_description->scanProject();
    return m_metaIO->load(d);
}

template <typename BaseIO>
ScanProjectPtr ScanProjectIO<BaseIO>::load() const
{
    ScanProjectPtr ret;

    // Load description and meta data for scan project
    Description d = m_baseIO->m_description->scanProject();

    // std::cout << "[HDF5IO - ScanProjectIO - load]: Description" << std::endl;
     std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    if(*d.dataRoot != "" && !m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        std::cout << "[ScanProjectIO] Warning: '" << *d.dataRoot << "' does not exist." << std::endl; 
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        try {
            ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
        } catch(const YAML::TypedBadConversion<ScanProject>& ex) {
            std::cerr << "[ScanProjectIO - load] ERROR at ScanProject: Could not decode YAML as ScanProject." << std::endl;
            throw ex;
        }
    } 
    else 
    {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        // std::cout << timestamp << "[ScanProjectIO] Could not load meta information. No meta name specified." << std::endl;
        ret = std::make_shared<ScanProject>();
    }


    // Get all sub scans
    size_t scanPosNo = 0;
    while(true)
    {  
        // std::cout << "[ScanProjectIO - load] try load ScanPosition "  << scanPosNo << std::endl;
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

template <typename BaseIO>
void ScanProjectIO<BaseIO>::saveScanProject(
    ScanProjectPtr scanProject) const
{
    save(scanProject);
}

template <typename BaseIO>
ScanProjectPtr ScanProjectIO<BaseIO>::loadScanProject() const
{
    return load();
}

template <typename BaseIO>
ScanProjectPtr ScanProjectIO<BaseIO>::loadScanProject(ReductionAlgorithmPtr reduction) const
{
    ScanProjectPtr ret;

    // Load description and meta data for scan project
    Description d = m_baseIO->m_description->scanProject();

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    if(*d.dataRoot != "" && !m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        std::cout << "[ScanProjectIO] Warning: '" << *d.dataRoot << "' does not exist." << std::endl; 
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);
        ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
    } 
    else 
    {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        // std::cout << timestamp << "[ScanProjectIO] Could not load meta information. No meta name specified." << std::endl;
        ret = std::make_shared<ScanProject>();
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

} // namespace scanio
} // namespace lvr2

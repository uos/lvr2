namespace lvr2
{
namespace scanio
{

template<typename BaseIO>
void ScanProjectIO<BaseIO>::save(
    ScanProjectPtr scanProject) const
{
    Description d = m_baseIO->m_description->scanProject();


//     lvr2::logout::get() << "[ScanProjectIO - save]: Description" << lvr2::endl;
//     lvr2::logout::get() << d << lvr2::endl;
//
//     if(!d.dataRoot)
//     {
//         lvr2::logout::get() << lvr2::info << "[ScanProjectIO] Description does not contain a data root for the ScanProject" << lvr2::endl;
//         d.dataRoot = "";
//     }
//
//     lvr2::logout::get() << "[ScanProjectIO] Save Scan Project "<< lvr2::endl;
    // Iterate over all positions and save
    size_t successfulScans = 1;

    // Starting at i=0 but successfulScans = 1 because scanProject->positions[0] is Scan 1
    for (size_t i = 0; i < scanProject->positions.size(); i++)
    {

        if (!(m_scanPositionIO->saveScanPosition(successfulScans, scanProject->positions[i]))){
            successfulScans--;

        }
        successfulScans++;
    }

    // Default scan project yaml
    if(d.meta)
    {
        YAML::Node node;
        node = *scanProject;
        // lvr2::logout::get() << "[ScanProjectIO] saveMetaYAML, Group: "
        //             << *d.groupName << ", metaName: " << *d.metaName << lvr2::endl;
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

    //lvr2::logout::get() << "[HDF5IO - ScanProjectIO - load]: Description" << lvr2::endl;
    //lvr2::logout::get() << d << lvr2::endl;

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    if(*d.dataRoot != "" && !m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        lvr2::logout::get() << lvr2::warning << "[ScanProjectIO] Warning: '" << *d.dataRoot << "' does not exist." << lvr2::endl; 
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        if (!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        try
        {
            ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
        }
        catch (const YAML::TypedBadConversion<ScanProject> &ex)
        {
            lvr2::logout::get() << lvr2::error << "[ScanProjectIO - load] ScanProject: Could not decode YAML as ScanProject." << lvr2::endl;
            throw ex;
        }
    } 
    else 
    {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        lvr2::logout::get() << lvr2::warning << "[ScanProjectIO] Could not load meta information. No meta name specified." << lvr2::endl;
        ret = std::make_shared<ScanProject>();
    }


    // Get all sub scans
    size_t scanPosNo = 0;
    while(true)
    {  
         lvr2::logout::get() << "[ScanProjectIO - load] try load ScanPosition "  << scanPosNo << lvr2::endl;
        // Get description for next scan

        // Generate Directories for all Filetypes
        ScanPositionPtr scanPos = m_scanPositionIO->loadScanPosition(scanPosNo);
        if(scanPos)
        {
            ret->positions.push_back(scanPos);
        }
        else if(scanPosNo > 1)
        {
            break;
        }

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
        lvr2::logout::get() << lvr2::warning << "[ScanProjectIO] " << *d.dataRoot << "' does not exist." << lvr2::endl; 
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

        // lvr2::logout::get() << lvr2::info << "[ScanProjectIO] Could not load meta information. No meta name specified." << lvr2::endl;
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
            //if
            break;
        }
        ret->positions.push_back(scanPos);
        scanPosNo++;
    }

    return ret;
}

} // namespace scanio
} // namespace lvr2

#include "lvr2/io/yaml/ScanProject.hpp"

namespace lvr2
{

template<typename FeatureBase>
void ScanProjectIO<FeatureBase>::save(
    ScanProjectPtr scanProject) const
{
    Description d = m_featureBase->m_description->scanProject();

    std::cout << "[ScanProjectIO] ScanProject - Description: " << std::endl;
    std::cout << d << std::endl;

    if(!d.groupName)
    {
        std::cout << timestamp << "[ScanProjectIO] Description does not contain a group for the ScanProject" << std::endl;
        d.groupName = "";
    }

    // Default scan project yaml
    if(d.metaName)
    {
        YAML::Node node;
        node = *scanProject;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }

    // std::cout << "[ScanProjectIO] Save Scan Project "<< std::endl;
    // Iterate over all positions and save
    for (size_t i = 0; i < scanProject->positions.size(); i++)
    {
        // std::cout << "[ScanProjectIO] Save Pos" << i << std::endl;
        m_scanPositionIO->saveScanPosition(i, scanProject->positions[i]);
    }
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
    ScanProjectPtr ret;

    // Load description and meta data for scan project
    Description d = m_featureBase->m_description->scanProject();

    if(!d.groupName)
    {
        d.groupName = "";
    }

    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    if(d.metaName)
    {
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<ScanProject>(meta.as<ScanProject>());
    } else {
        // what to do here?
        // some schemes do not have meta descriptions: slam6d
        // create without meta information: generate meta afterwards

        std::cout << timestamp << "[ScanProjectIO] Could not load meta information. No meta name specified." << std::endl;
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
        ret->positions.push_back(scanPos);
        scanPosNo++;
    }

    return ret;
}

} // namespace lvr2

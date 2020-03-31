#include "lvr2/io/yaml/ScanProject.hpp"

namespace lvr2
{

template <typename FeatureBase>
void ScanProjectIO<FeatureBase>::saveScanProject(const ScanProjectPtr& scanProjectPtr)
{
    Description d = m_featureBase->m_description.scanProject();

    std::string group = "";
    std::string metaName = "meta.yaml";

    if(d.groupName)
    {
        group = *d.groupName;
    }

    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    // Check for meta data and save
    if(d.metaData)
    {
        m_featureBase->m_kernel.saveMetaYAML(group, metaName, *d.metaData);
    }
    else
    {
        // Create default meta and save
        YAML::Node node;
        node[""] = (ScanProject)(*(scanProjectPtr));
        m_featureBase->m_kernel.saveMetaYAML(group, metaName, node);
    }   
    
    // Iterate over all positions and save
    for (size_t i = 0; i < scanProjectPtr->positions.size(); i++)
    {
        m_scanPositionIO->saveScanPosition(i, scanProjectPtr->positions[i]);
    }
}

template <typename FeatureBase>
ScanProjectPtr ScanProjectIO<FeatureBase>::loadScanProject()
{
    ScanProjectPtr ret(new ScanProject);
    ScanProject* project = new ScanProject;
    // Load description and meta data for scan project
    Description d = m_featureBase->m_description.scanProject();
    if(d.metaData)
    {
        *ret = (d.metaData.get()).as<ScanProject>();
    }

    // Get all sub scans
    size_t scanPosNo = 0;
    do
    {
        // Get description for next scan
        Description scanDescr = m_featureBase->m_description.position(scanPosNo);

        // Check if it exists. If not, exit.
        if(m_featureBase->m_kernel.exists(*scanDescr.groupName))
        {
            std::cout << timestamp 
                      << "ScanPositionIO: Loading scanposition " 
                      << scanPosNo << std::endl;
                  
            ScanPositionPtr scanPos = m_scanPositionIO->loadScanPosition(scanPosNo);
            ret->positions.push_back(scanPos);
        }
        else
        {
            break;
        }
        ++scanPosNo;
    } 
    while (true);

    return ret;
}


} // namespace lvr2

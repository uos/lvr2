#include "lvr2/io/yaml/ScanProject.hpp"

namespace lvr2
{

template <typename FeatureBase>
void ScanProjectIO<FeatureBase>::saveScanProject(const ScanProjectPtr& scanProjectPtr)
{
    Description d = m_featureBase->m_description->scanProject();

    // Default names
    std::string group = "";
    std::string metaName = "meta.yaml";

    // Default scan project yaml
    YAML::Node node;
    node = *scanProjectPtr;

    // Try to override defaults
    if(d.groupName)
    {
        group = *d.groupName;
    }

    if(d.metaName)
    {
        node = *d.metaName;
    }
    m_featureBase->m_kernel->saveMetaYAML(group, metaName, node);
    
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

    // Load description and meta data for scan project
    Description d = m_featureBase->m_description->scanProject();
    if(d.metaData)
    {
        try
        {
            *ret = (d.metaData.get()).as<ScanProject>();
        }
        catch(YAML::TypedBadConversion<ScanProject>& e)
        {
            d.metaData = boost::none;  
        }
    }

    // Get all sub scans
    size_t scanPosNo = 0;
    do
    {
        // Get description for next scan
        Description scanDescr = m_featureBase->m_description->position(scanPosNo);

        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", scanDescr);

        // Check if scan position group is valid, else break
        if(scanDescr.groupName)
        {
            // Check if it exists. If not, exit.
            if (m_featureBase->m_kernel->exists(groupName))
            {
                std::cout << timestamp
                          << "ScanProjectIO: Loading scanposition "
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
        else
        {
            break;
        }
    } 
    while (true);

    return ret;
}


} // namespace lvr2

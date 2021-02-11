#include "lvr2/io/yaml/LIDAR.hpp"

namespace lvr2 {

template <typename FeatureBase>
void LIDARIO<FeatureBase>::save(
    const size_t& scanPosNr,
    const size_t& lidarNr,
    LIDARPtr lidar) const
{
    
    auto D = m_featureBase->m_description;
    Description d = D->lidar(D->position(scanPosNr), lidarNr);

    // std::cout << "[LIDARIO] LIDAR - Description: " << std::endl;
    // std::cout << d << std::endl;
    
    if(d.metaName)
    {
        YAML::Node node;
        node = *lidar;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }

    // std::cout << "[LIDARIO] Meta written. " << std::endl;

    // Save all scans of lidar
    for(size_t scanNr = 0; scanNr < lidar->scans.size(); scanNr++)
    {
        m_scanIO->save(scanPosNr, lidarNr, scanNr, lidar->scans[scanNr]);
    }

    // std::cout << "[LIDARIO] Scans written. " << std::endl;
}

template <typename FeatureBase>
LIDARPtr LIDARIO<FeatureBase>::load(
    const size_t& scanPosNr,
    const size_t& lidarNr) const
{
    LIDARPtr ret;

    auto D = m_featureBase->m_description;
    Description d = D->lidar(D->position(scanPosNr), lidarNr);

    // check if group exists
    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    // std::cout << "[LIDARIO - load] Description:" << std::endl;
    // std::cout << d << std::endl;

    if(d.metaName)
    {
        if(!m_featureBase->m_kernel->exists(*d.groupName, *d.metaName))
        {
            // std::cout << timestamp << " [LIDARIO]: Specified meta file not found. " << std::endl;
            return ret;
        } 

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<LIDAR>(meta.as<LIDAR>());
        
    } else {
        
        // no meta name specified but scan position is there: 
        ret.reset(new LIDAR);
    }

    size_t scanNr = 0;
    while(true)
    {
        ScanPtr scan = m_scanIO->load(scanPosNr, lidarNr, scanNr);
        if(scan)
        {
            ret->scans.push_back(scan);
        } else {
            break;
        }
        scanNr++;
    }
    
}


} // namespace lvr2
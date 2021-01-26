#include "lvr2/io/yaml/Scan.hpp"
#include <boost/optional/optional_io.hpp>
#include "lvr2/registration/OctreeReduction.hpp"

namespace lvr2
{

template <typename FeatureBase>
void ScanIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    auto Dgen = m_featureBase->m_description;
    Description pos_descr = Dgen->position(scanPosNo);
    Description lidar_descr = Dgen->lidar(pos_descr, sensorNo);    
    Description d = Dgen->scan(lidar_descr, scanNo);

    std::cout << "[ScanIO] Scan " << scanPosNo << "," << scanNo <<  " - Description: " << std::endl;
    std::cout << d << std::endl;

    if(d.metaName)
    {
        YAML::Node node;
        node = *scanPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }

    if(!d.dataSetName)
    {
        // Scan is not a dataset: handle as group of channels
        m_pclIO->savePointCloud(*d.groupName, scanPtr->points);
    } else {
        // Scan is a dataset: write data
        m_pclIO->savePointCloud(*d.groupName, *d.dataSetName, scanPtr->points);
    }
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::load(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    ScanPtr ret;

    // Get Description of Scan Location

    auto Dgen = m_featureBase->m_description;
    Description pos_descr = Dgen->position(scanPosNo);
    Description lidar_descr = Dgen->lidar(pos_descr, sensorNo);    
    Description d = Dgen->scan(lidar_descr, scanNo);

    if(!d.groupName)
    {
        return ret;
    }

    if(d.metaName)
    {
        if(!m_featureBase->m_kernel->exists(*d.groupName, *d.metaName))
        {
            return ret;
        }

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<Scan>(meta.as<Scan>());
    } else {
        // for schemas without meta information
        ret.reset(new Scan);
    }

    if(d.dataSetName)
    {
        // Load actual data
        ret->points = m_pclIO->loadPointCloud(*d.groupName, *d.dataSetName);
    } else {
        ret->points = m_pclIO->loadPointCloud(*d.groupName);
    }

    return ret;
}

template <typename FeatureBase>
void ScanIO<FeatureBase>::saveScan(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    save(scanPosNo, scanNo, scanPtr);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    return load(scanPosNo, sensorNo, scanNo);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo, 
    ReductionAlgorithmPtr reduction) const
{
    ScanPtr ret = loadScan(scanPosNo, scanNo);

    if(ret)
    {
        if(ret->points)
        {
            reduction->setPointBuffer(ret->points);
            ret->points = reduction->getReducedPoints();
        }
    }

    return ret;
}

} // namespace lvr2

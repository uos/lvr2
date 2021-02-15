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
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    //// DATA
    if(!d.data)
    {
        // Scan is not a dataset: handle as group of channels
        m_pclIO->save(scanPosNo, sensorNo, scanNo, scanPtr->points);
    } else {
        // Scan is a dataset: write data
        m_pclIO->save(*d.dataRoot, *d.data, scanPtr->points);
    }

    //// META
    if(d.meta)
    {
        YAML::Node node;
        node = *scanPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
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
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    if(!d.dataRoot)
    {
        return ret;
    }

    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }


    std::cout << "[ScanIO - load] Description:" << std::endl;
    std::cout << d << std::endl;

    /// META

    if(d.meta)
    {
        if(!m_featureBase->m_kernel->exists(*d.metaRoot, *d.meta))
        {
            return ret;
        }

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);
        ret = std::make_shared<Scan>(meta.as<Scan>());
    } else {
        // for schemas without meta information
        ret.reset(new Scan);
    }


    /// DATA
    // if(d.data)
    // {
    //     // Load actual data
    //     ret->points = m_pclIO->load(*d.dataRoot, *d.data);
    // } else {
    //     ret->points = m_pclIO->load(*d.dataRoot);
    // }

    // std::cout << "Loaded: " << ret->points->numPoints() << std::endl;

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

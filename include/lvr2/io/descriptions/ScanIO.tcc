#include "lvr2/io/yaml/Scan.hpp"
#include <boost/optional/optional_io.hpp>
#include "lvr2/registration/OctreeReduction.hpp"

namespace lvr2
{

template <typename FeatureBase>
void ScanIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    // Default meta yaml

    // Get group and dataset names according to 
    // data fomat description and override defaults if 
    // when possible
    Description d = m_featureBase->m_description->scan(scanPosNo, scanNo);

    // std::cout << "[ScanIO] Scan " << scanPosNo << "," << scanNo <<  " - Description: " << std::endl;
    // std::cout << d << std::endl;

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

    // Save Waveform data
    if (scanPtr->waveform)
    {
	    // std::cout << "[ScanIO]Waveform found " <<std::endl;
        m_fullWaveformIO->saveFullWaveform(scanPosNo, scanNo, scanPtr->waveform);
    }
}

template <typename FeatureBase>
void ScanIO<FeatureBase>::saveScan(
    const size_t& scanPosNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    save(scanPosNo, scanNo, scanPtr);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo, const size_t& scanNo) const
{
    ScanPtr ret;

    // Get Description of Scan Location

    
    Description d = m_featureBase->m_description->scan(scanPosNo, scanNo);
    // std::cout << "[IO: ScanIO - load]: Description" << std::endl;
    // std::cout << d << std::endl;

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

    
    Description waveformDescr = m_featureBase->m_description->waveform(scanPosNo, scanNo);
    if(waveformDescr.dataSetName)
    {
        std::string dataSetName;
        std::string groupName;
        std::tie(groupName, dataSetName) = getNames("", "", waveformDescr);

        if (m_featureBase->m_kernel->exists(groupName))
        {
            std::cout << "[ScanIO] Loading Waveform" << std::endl;
            WaveformPtr fwPtr = m_fullWaveformIO->loadFullWaveform(scanPosNo, scanNo);
            ret->waveform = fwPtr;
            //boost::shared_array<uint16_t> waveformData(new uint16_t[fwPtr->waveformSamples.size()]);
            //std::memcpy(waveformData.get(), fwPtr->waveformSamples.data(), fwPtr->waveformSamples.size() * sizeof(uint16_t));
            //Channel<uint16_t>::Ptr waveformChannel(new Channel<uint16_t>(fwPtr->waveformSamples.size() / fwPtr->maxBucketSize, static_cast<size_t>(fwPtr->maxBucketSize), waveformData));
            //ret->points->addChannel<uint16_t>(waveformChannel, "waveform");
        } else{
            // std::cout << "[ScanIO] No Waveform found" << groupName << std::endl;
        }
    }

    // Change line above to this:
    // Let fullWaveformIO handle if waveform data exists
    // WaveformPtr fwPtr = m_fullWaveformIO->loadFullWaveform(scanPosNo, scanNo);
    // if(fwPtr)
    // {
    //     if(!ret){ ret.reset(new Scan); }
    //     ret->waveform = fwPtr;
    // }



    return ret;
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo, 
    const size_t& scanNo, 
    ReductionAlgorithmPtr reduction) const
{
    ScanPtr ret = loadScan(scanPosNo, scanNo);

    if(ret)
    {
        if(ret->points)
        {
            // check if given reduction is AllReductionAlgorithm
            AllReductionAlgorithm* allReductionPtr = dynamic_cast<AllReductionAlgorithm*> (reduction.get()); 
            if(allReductionPtr != nullptr)
            {        
                // in this case skip setPointBuffer()
                ret->points =  reduction->getReducedPoints();
            } 
            else 
            {
                reduction->setPointBuffer(ret->points);
                ret->points = reduction->getReducedPoints();
            }
        }
    }

    return ret;
}


template <typename FeatureBase>
bool ScanIO<FeatureBase>::isScan(const std::string& name)
{
    return true;
}

} // namespace lvr2

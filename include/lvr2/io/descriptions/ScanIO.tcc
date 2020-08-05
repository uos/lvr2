#include "lvr2/io/yaml/Scan.hpp"
#include <boost/optional/optional_io.hpp>

namespace lvr2
{

template <typename FeatureBase>
void ScanIO<FeatureBase>::saveScan(const size_t& scanPosNo, const size_t& scanNo, const ScanPtr& scanPtr)
{
    // Setup defaults: no group and scan number into .ply file. 
    // Write meta into a yaml file with same file name as the 
    // scan file. 
    std::string groupName = "";
   
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    std::string scanName = sstr.str() + ".ply";
    std::string metaName = sstr.str() + ".yaml";

    // Default meta yaml
    YAML::Node node;
    node = *scanPtr;

    // Get group and dataset names according to 
    // data fomat description and override defaults if 
    // when possible
    Description d = m_featureBase->m_description->scan(scanPosNo, scanNo);

    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    if(d.dataSetName)
    {
        scanName = *d.dataSetName;
    }

    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    cout << "SCAN META: " << d.metaData << endl;

    if(d.metaData)
    {
        node = *d.metaData;
    }

    // Save all scan data and meta data if present
    m_featureBase->m_kernel->savePointBuffer(groupName, scanName, scanPtr->points);
    
    // Get meta data from scan and save
    std::cout << "[SCAN IO] YAML save" << std::endl;
    //m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);

    // Save Waveform data
    if (scanPtr->waveform)
    {
	    std::cout << "[ScanIO]Waveform found " <<std::endl;
        m_fullWaveformIO->saveFullWaveform(scanPosNo, scanNo, scanPtr->waveform);
    } else 
    {
	    std::cout << "[ScanIO]no Waveform " <<std::endl;
    }

}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(const size_t& scanPosNo, const size_t& scanNo)
{
    ScanPtr ret(new Scan);

    Description d = m_featureBase->m_description->scan(scanPosNo, scanNo);

        std::cout << d.groupName << " " << d.dataSetName << std::endl; 

    // Init default values
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    std::string scanName = sstr.str() + ".ply";
    std::string metaName = sstr.str() + ".yaml";
    std::string groupName = "";

    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    if(d.dataSetName)
    {
        scanName = *d.dataSetName;
    }

    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    // Important! First load meta data as YAML cpp seems to 
    // create a new scan object before calling decode() !!!
    // Cf. https://stackoverflow.com/questions/50807707/yaml-cpp-encoding-decoding-pointers
    if(d.metaData)
    {
        *ret = (*d.metaData).as<Scan>();
    }
    else
    {
        std::cout << timestamp << "ScanIO::load(): Warning: No meta data found for "
                  << groupName << "/" << scanName << "." << std::endl;
    }

    // Load actual data
    ret->points = m_featureBase->m_kernel->loadPointBuffer(groupName, scanName);
    std::cout << groupName << " " << scanName << " <--------" << std::endl;
 
    // Get Waveform data
    Description waveformDescr = m_featureBase->m_description->waveform(scanPosNo, scanNo);
    if(waveformDescr.dataSetName)
    {
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", waveformDescr);

        if (m_featureBase->m_kernel->exists(groupName))
        {
            WaveformPtr fwPtr = m_fullWaveformIO->loadFullWaveform(scanPosNo, scanNo);
            ret->waveform = fwPtr;
        }
    }


    return ret;
}

// template <typename FeatureBase>
// ScanPtr ScanIO<FeatureBase>::load(const std::string& group, const std::string& name)
// {
//     ScanPtr ret;
//     return ret;
// }


template <typename FeatureBase>
bool ScanIO<FeatureBase>::isScan(const std::string& name)
{
    return true;
}

} // namespace lvr2

#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/yaml/Waveform.hpp"
#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{


template <typename Derived>
void FullWaveformIO<Derived>::saveFullWaveform(
    const size_t& scanPosNo,
    const size_t& scanNo,
    const WaveformPtr& fwPtr)
{
    Description d = m_featureBase->m_description->waveform(scanPosNo, scanNo);
    // Init default values
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    std::string waveformName = sstr.str() + ".lwf";
    std::string metaName = sstr.str() + ".yaml";
    std::string groupName = "";
    
    std::cout << "Saving Waveform to " << scanPosNo << " " << scanNo << std::endl;
    // Default meta yaml
    YAML::Node node;
    node = *fwPtr;

    // Get group and dataset names according to 
    // data fomat description and override defaults if 
    // when possible
    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    if(d.dataSetName)
    {
        waveformName = *d.dataSetName;
    }

    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    if(d.metaData)
    {
        node = *d.metaData;
    }
    
    m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
    
    // saving Waveform samples
    std::vector<size_t> waveformDim = {fwPtr->waveformSamples.size() / fwPtr->maxBucketSize, static_cast<size_t>(fwPtr->maxBucketSize)};
    uint16Arr waveformData = uint16Arr(new uint16_t[fwPtr->waveformSamples.size()]);
    std::memcpy(waveformData.get(), fwPtr->waveformSamples.data(), fwPtr->waveformSamples.size() * sizeof(uint16_t));
    m_featureBase->m_kernel->saveUInt16Array(groupName, waveformName, waveformDim, waveformData);
    std::cout << "Waveform saved" << std::endl; 

}

template <typename Derived>
WaveformPtr FullWaveformIO<Derived>::loadFullWaveform(const size_t& scanPosNo, const size_t& scanNo)
{
    WaveformPtr ret(new Waveform);
    Description d = m_featureBase->m_description->waveform(scanPosNo, scanNo);
    // Init default values
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    std::string waveformName = sstr.str() + ".lwf";
    std::string metaName = sstr.str() + ".yaml";
    std::string groupName = "";

    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    if(d.dataSetName)
    {
        waveformName = *d.dataSetName;
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
        *ret = (*d.metaData).as<Waveform>();
    }
    else
    {
        std::cout << timestamp << "WaveformIO::load(): Warning: No meta data found for "
                  << groupName << "/" << waveformName << "." << std::endl;
    }

    // Load actual data
    boost::shared_array<uint16_t> waveformData;
    std::vector<size_t> waveformDim;
    waveformData = m_featureBase->m_kernel->loadUInt16Array(groupName, waveformName, waveformDim);
    std::cout << " Dims ..>" << waveformDim[0] << " " <<  waveformDim[1] << std::endl;
    ret->waveformSamples = std::vector<uint16_t>(waveformData.get(), waveformData.get() + (waveformDim[0] * waveformDim[1]));
    ret->maxBucketSize = waveformDim[1];
    return ret;

  
}



} // namespace lvr2

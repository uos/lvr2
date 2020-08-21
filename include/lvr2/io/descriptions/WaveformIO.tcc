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
    std::string groupName = "";
   
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    std::string waveformName = sstr.str() + ".lwf";
    std::string metaName = sstr.str() + ".yaml";

    // Default meta yaml
    YAML::Node node;
    node = *fwPtr;

    // Get group and dataset names according to 
    // data fomat description and override defaults if 
    // when possible
 
    Description d = m_featureBase->m_description->waveform(scanPosNo, scanNo);
   

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
std::cout << "Saving Waveform to " << groupName << std::endl;
    m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
    // saving Waveform samples
    //m_matrixIO->saveMatrix(groupName, waveformName, fwPtr->waveformSamples);

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
    //ret->waveformSamples = m_matrixIO->template loadMatrix<uint16_t>(groupName, waveformName);

    boost::shared_array<uint16_t> waveformData;
    std::vector<size_t> waveformDim;
    std::cout << "loading " << groupName << " / " << waveformName <<  std::endl;
    waveformData = m_featureBase->m_kernel->loadUInt16Array(groupName, waveformName, waveformDim);
    //std::cout << "Waveform MAtrix " << waveformDim[0] << " "  << std::endl;
    //ret->waveformSamples = m_featureBase->m_kernel->loadArray<uint16_t>
    //TODO: THE REST
    return ret;

  
}



} // namespace lvr2

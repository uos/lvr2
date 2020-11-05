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
    uint16Arr waveformData = uint16Arr(new uint16_t[fwPtr->lowPower.size() * (fwPtr->maxBucketSize + 1)]());
    for(int i = 0; i < fwPtr->lowPower.size(); i++)
    {
        waveformData[(fwPtr->maxBucketSize + 1) * i] = (fwPtr->lowPower[i] ? 1 : 0);
        //uint16_t* startOfBlock = fwPtr->waveformSample.data()
        int sizeOfBlock = fwPtr->waveformIndices[i + 1] - fwPtr->waveformIndices[i];
        std::memcpy(waveformData.get() + (i * (fwPtr->maxBucketSize)) + 1, fwPtr->waveformSamples.data() + fwPtr->waveformIndices[i], sizeOfBlock * sizeof(uint16_t));
    }

    std::vector<size_t> waveformDim = {fwPtr->lowPower.size(), static_cast<size_t>(fwPtr->maxBucketSize)};
    m_featureBase->m_kernel->saveUInt16Array(groupName, waveformName, waveformDim, waveformData);

}
template <typename Derived>
void FullWaveformIO<Derived>::saveLabelWaveform(
    const std::string& groupName,
    const WaveformPtr& fwPtr)
{
    std::cout << fwPtr->maxBucketSize << std::endl;
    std::cout << fwPtr->waveformIndices.size() << std::endl;
    // saving Waveform samples
    uint16Arr waveformData = uint16Arr(new uint16_t[fwPtr->lowPower.size() * (fwPtr->maxBucketSize + 1)]());
    for(int i = 0; i < fwPtr->lowPower.size(); i++)
    {
        waveformData[(fwPtr->maxBucketSize + 1) * i] = (fwPtr->lowPower[i] ? 1 : 0);
        int sizeOfBlock = fwPtr->waveformIndices[i + 1] - fwPtr->waveformIndices[i];
        std::memcpy(waveformData.get() + (i * (fwPtr->maxBucketSize)) + 1, fwPtr->waveformSamples.data() + fwPtr->waveformIndices[i], sizeOfBlock * sizeof(uint16_t));
    }

    std::vector<size_t> waveformDim = {fwPtr->lowPower.size(), static_cast<size_t>(fwPtr->maxBucketSize)};
    m_featureBase->m_kernel->saveUInt16Array(groupName, "waveform", waveformDim, waveformData);

}
template <typename Derived>
WaveformPtr FullWaveformIO<Derived>::loadLabelWaveform(const std::string& groupName)
{
    WaveformPtr ret(new Waveform);
    std::string waveformName = "waveform";

    // Load actual data
    boost::shared_array<uint16_t> waveformData;
    std::vector<size_t> waveformDim;
    waveformData = m_featureBase->m_kernel->loadUInt16Array(groupName, waveformName, waveformDim);
    ret->waveformSamples.reserve(waveformDim[0] * waveformDim[1] - 1);
    ret->lowPower.reserve(waveformDim[0]);
    ret->waveformIndices.reserve(waveformDim[0]);

    //Store first entry
    ret->waveformIndices.push_back(0);

    //Remove entries containing 0 from the data and calculate the iterators

    for (int i = 0; i < waveformDim[0]; i++)
    {
        //First entry is the channel
        ret->lowPower.push_back(waveformData[i * waveformDim[1]]);

        //Start with 1 since the first entry was channelinfo
        for(int j = 1; j < waveformDim[1]; j++)
        {
            if(waveformData[(i * waveformDim[1]) + j] == 0 || j == (waveformDim[1] - 1))
            {
                if(waveformData[(i * waveformDim[1]) + j] != 0 )
                {
                    ret->waveformSamples.push_back(waveformData[(i * waveformDim[1]) + j]);
                    ret->waveformIndices.push_back(ret->waveformIndices.back() + j);
                }
                else
                {
                    ret->waveformIndices.push_back(ret->waveformIndices.back() + j - 1);
                }
                break;
            }
            ret->waveformSamples.push_back(waveformData[(i * waveformDim[1]) + j]);
        }
    }

    ret->maxBucketSize = waveformDim[1] - 1;
    return ret;
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
    ret->waveformSamples.reserve(waveformDim[0] * waveformDim[1] - 1);
    ret->lowPower.reserve(waveformDim[0]);
    ret->waveformIndices.reserve(waveformDim[0]);
    //Store first entry
    ret->waveformIndices.push_back(0);

    //Remove entries containing 0 from the data and calculate the iterators

    for (int i = 0; i < waveformDim[0]; i++)
    {
        //First entry is the channel
        ret->lowPower.push_back(waveformData[i * waveformDim[1]]);

        //Start with 1 since the first entry was channelinfo
        for(int j = 1; j < waveformDim[1]; j++)
        {
            if(waveformData[(i * waveformDim[1]) + j] == 0 || j == (waveformDim[1] - 1))
            {
                if(waveformData[(i * waveformDim[1]) + j] != 0 )
                {
                    ret->waveformSamples.push_back(waveformData[(i * waveformDim[1]) + j]);
                    ret->waveformIndices.push_back(ret->waveformIndices.back() + j);
                }
                else
                {
                    //ret->waveformSamples.insert(ret->waveformSamples.end(), waveformData[(i * waveformDim[1]) + 1], waveformData[(i * waveformDim[1]) + 1 + j - 1]);
                    ret->waveformIndices.push_back(ret->waveformIndices.back() + j - 1);
                }
                break;
            }
            ret->waveformSamples.push_back(waveformData[(i * waveformDim[1]) + j]);
        }
    }

    //ret->waveformSamples->shrink_to_fit();
    //ret->waveformSamples = std::vector<uint16_t>(waveformData.get(), waveformData.get() + (waveformDim[0] * waveformDim[1]));
    ret->maxBucketSize = waveformDim[1] - 1;
    return ret;

  
}



} // namespace lvr2

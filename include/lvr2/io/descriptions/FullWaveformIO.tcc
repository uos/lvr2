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
    sstr << "scan" << std::setfill('0') << std::setw(8) << scanNo;
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

    m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
    // saving Waveform samples
    //m_matrixIO->saveMatrix(groupName, waveformName, fwPtr->waveform);

    std::vector<size_t> dim = {fwPtr->amplitude.size()};
    // saving amplitude
    floatArr amplitude(new float[fwPtr->amplitude.size()]);
    std::memcpy(&amplitude, fwPtr->amplitude.data(), fwPtr->amplitude.size());
    m_arrayIO->saveFloatArray(groupName, "amplitude", dim, amplitude);

    // saving deviation
    dim = {fwPtr->deviation.size()};
    floatArr deviation(new float[fwPtr->deviation.size()]);
    std::memcpy(&deviation, fwPtr->deviation.data(), fwPtr->deviation.size());
    m_arrayIO->saveFloatArray(groupName, "deviation", dim, deviation);

    // saving reflectance
    dim = {fwPtr->reflectance.size()};
    floatArr reflectance(new float[fwPtr->reflectance.size()]);
    std::memcpy(&reflectance, fwPtr->reflectance.data(), fwPtr->reflectance.size());
    m_arrayIO->saveFloatArray(groupName, "reflectance", dim, reflectance);

   
    // saving backgroundRadiation
    dim = {fwPtr->backgroundRadiation.size()};
    floatArr backgroundRadiation(new float[fwPtr->backgroundRadiation.size()]);
    std::memcpy(&backgroundRadiation, fwPtr->backgroundRadiation.data(), fwPtr->backgroundRadiation.size());
    m_arrayIO->saveFloatArray(groupName, "backgroundRadiation", dim, backgroundRadiation);

}

template <typename Derived>
WaveformPtr FullWaveformIO<Derived>::loadFullWaveform(const size_t& scanPosNo, const size_t& scanNo)
{
    WaveformPtr ret(new Waveform);
    Description d = m_featureBase->m_description->waveform(scanPosNo, scanNo);
    // Init default values
    std::stringstream sstr;
    sstr << "scan" << std::setfill('0') << std::setw(8) << scanNo;
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
    //ret->waveform = m_matrixIO->loadMatrix(groupName, waveformName);
   
    //TODO: THE REST
    return ret;

   /* 
    // Default path
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    std::string groupName = "raw/" + sstr.str() + "/spectral/data";

    floatArr amplitude;
    flaotArr reflectance;
    floatArr backgroudnRadiation;
    floatArr deviation;


    m_featureBase->m_kernel->subGroupNames(pointCl, std::regex("\\d{8}"), positionGroups);

    for (std::string positionGroup : positionGroups)
    {
        Description fd = m_featureBase->m_description->hyperSpectralFrames(positionGroup);
        
        ucharArr data;
        doubleArr timestamps;
        std::vector<size_t> dim; // Uff, initialisierung???
        if(fd.dataSetName)
        {
            data = m_arrayIO->loadUCharArray(positionGroup, *fd.dataSetName, dim);
        }
        
        std::vector<size_t> timeDim;
        if(td.dataSetName)
        {   
            timestamps = m_arrayIO->loadDoubleArray(positionGroup, *td.dataSetName, timeDim);
        }

        HyperspectralPanoramaPtr panoramaPtr(new HyperspectralPanorama);
        if(data && timestamps)
        {
            for (int i = 0; i < dim[0]; i++)
            {
                // img size ist dim[1] * dim[2]

                cv::Mat img = cv::Mat(dim[1], dim[2], CV_8UC1);
                std::memcpy(
                    img.data, data.get() + i * dim[1] * dim[2], dim[1] * dim[2] * sizeof(uchar));

                HyperspectralPanoramaChannelPtr channelPtr(new HyperspectralPanoramaChannel);
                channelPtr->channel = img;
                channelPtr->timestamp = timestamps[i];
                panoramaPtr->channels.push_back(channelPtr);
            }
        }
        else
        {
            if(!data)
            {
                std::cout << timestamp 
                          << "HypersprectralCameraIO::load() Warning: No image data found: " 
                          << *fd.dataSetName << std::endl;
            }
            if(!timestamps)
            {
                std::cout << timestamp 
                          << "HypersprectralCameraIO::load() Warning: No timestamps found: " 
                          << *td.dataSetName << std::endl;
            }
        }
        ret->panoramas.push_back(panoramaPtr);
    }
    return ret;
*/
}



} // namespace lvr2

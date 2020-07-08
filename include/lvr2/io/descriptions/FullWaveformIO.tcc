#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{


template <typename Derived>
void FullWaveformIO<Derived>::saveFullWaveform(
    const size_t& scanPosNo,
    const size_t& scanNo,
    const FullWaveformPtr& fwPtr)
{
    std::string id(FullWaveformIO<Derived>::ID);
    std::string obj(FullWaveformIO<Derived>::OBJID);
  
    Description d = m_featureBase->m_description->position(scanPosNo);
    
    // Setup default string for scan position
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    std::string group = sstr.str();

    // Override if descriptions contains a position string
    if(d.groupName)
    {
        group = *d.groupName;
    }

    //  hdf5util::setAttribute(group, "IO", id);
    //  hdf5util::setAttribute(group, "CLASS", obj);

    // saving Waveform samples
    //m_matrixIO->saveMatrix(group, "waveform", fwPtr->waveform);

    std::vector<size_t> dim = {fwPtr->amplitude.size()};
    // saving amplitude
    floatArr amplitude(new float[fwPtr->amplitude.size()]);
    std::memcpy(&amplitude, fwPtr->amplitude.data(), fwPtr->amplitude.size());
    m_arrayIO->saveFloatArray(group, "amplitude", dim, amplitude);

    // saving deviation
    dim = {fwPtr->deviation.size()};
    floatArr deviation(new float[fwPtr->deviation.size()]);
    std::memcpy(&deviation, fwPtr->deviation.data(), fwPtr->deviation.size());
    m_arrayIO->saveFloatArray(group, "deviation", dim, deviation);

    // saving reflectance
    dim = {fwPtr->reflectance.size()};
    floatArr reflectance(new float[fwPtr->reflectance.size()]);
    std::memcpy(&reflectance, fwPtr->reflectance.data(), fwPtr->reflectance.size());
    m_arrayIO->saveFloatArray(group, "reflectance", dim, reflectance);

   
    // saving backgroundRadiation
    dim = {fwPtr->backgroundRadiation.size()};
    floatArr backgroundRadiation(new float[fwPtr->backgroundRadiation.size()]);
    std::memcpy(&backgroundRadiation, fwPtr->backgroundRadiation.data(), fwPtr->backgroundRadiation.size());
    m_arrayIO->saveFloatArray(group, "backgroundRadiation", dim, backgroundRadiation);

}

template <typename Derived>
FullWaveformPtr FullWaveformIO<Derived>::loadFullWaveform(const size_t& scanPosNo, const size_t& scanNo)
{
    FullWaveformPtr ret(new FullWaveform);
    Description d = m_featureBase->m_description->fullWaveform(scanPosNo, scanNo);
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
*/
    return ret;
}



// template <typename Derived>
// HyperspectralCameraPtr HyperspectralCameraIO<Derived>::load(HighFive::Group& group)
// {
//     HyperspectralCameraPtr ret(new HyperspectralCamera);

//     if (!isHyperspectralCamera(group))
//     {
//         std::cout << "[Hdf5IO - HyperspectralCameraIO] WARNING: flags of " << group.getId()
//                   << " are not correct." << std::endl;
//         return ret;
//     }

//     // read extrinsics
//     boost::optional<lvr2::Extrinsicsd> extrinsics =
//         m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsics");
//     if (extrinsics)
//     {
//         ret->extrinsics = extrinsics.get();
//     }

//     // read extrinsicsEstimate
//     boost::optional<lvr2::Extrinsicsd> extrinsicsEstimate =
//         m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsicsEstimate");
//     if (extrinsicsEstimate)
//     {
//         ret->extrinsicsEstimate = extrinsicsEstimate.get();
//     }

//     // read focalLength
//     if (group.exist("focalLength"))
//     {
//         std::vector<size_t> dimension;
//         doubleArr focalLength = m_arrayIO->template load<double>(group, "focalLength", dimension);

//         if (dimension.at(0) != 1)
//         {
//             std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong focalLength dimension. The focalLength "
//                          "will not be loaded."
//                       << std::endl;
//         }
//         else
//         {
//             ret->focalLength = focalLength[0];
//         }
//     }

//     // read offsetAngle
//     if (group.exist("offsetAngle"))
//     {
//         std::vector<size_t> dimension;
//         doubleArr offsetAngle = m_arrayIO->template load<double>(group, "offsetAngle", dimension);

//         if (dimension.at(0) != 1)
//         {
//             std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong offsetAngle dimension. The offsetAngle "
//                          "will not be loaded."
//                       << std::endl;
//         }
//         else
//         {
//             ret->offsetAngle = offsetAngle[0];
//         }
//     }

//     // read principal
//     if (group.exist("principal"))
//     {
//         std::vector<size_t> dimension;
//         doubleArr principal = m_arrayIO->template load<double>(group, "principal", dimension);

//         if (dimension.at(0) != 3)
//         {
//             std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong principal dimension. The principal "
//                          "will not be loaded."
//                       << std::endl;
//         }
//         else
//         {
//             Vector3d p = {principal[0], principal[1], principal[2]};
//             ret->principal = p;
//         }
//     }

//     // read distortion
//     if (group.exist("distortion"))
//     {
//         std::vector<size_t> dimension;
//         doubleArr distortion = m_arrayIO->template load<double>(group, "distortion", dimension);

//         if (dimension.at(0) != 3)
//         {
//             std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong distortion dimension. The distortion "
//                          "will not be loaded."
//                       << std::endl;
//         }
//         else
//         {
//             Vector3d d = {distortion[0], distortion[1], distortion[2]};
//             ret->distortion = d;
//         }
//     }

//     // iterate over all panoramas
//     for (std::string groupname : group.listObjectNames())
//     {
//         // load all scanCameras
//         if (std::regex_match(groupname, std::regex("\\d{8}")))
//         {
//             HighFive::Group g = hdf5util::getGroup(group, "/" + groupname);

//             std::vector<size_t> dim;
//             ucharArr data = m_arrayIO->template load<uchar>(g, "frames", dim);

//             std::vector<size_t> timeDim;
//             doubleArr timestamps = m_arrayIO->template load<double>(g, "timestamps", timeDim);

//             HyperspectralPanoramaPtr panoramaPtr(new HyperspectralPanorama);
//             for (int i = 0; i < dim[0]; i++)
//             {
//                 // img size ist dim[1] * dim[2]

//                 cv::Mat img = cv::Mat(dim[1], dim[2], CV_8UC1);
//                 std::memcpy(
//                     img.data, data.get() + i * dim[1] * dim[2], dim[1] * dim[2] * sizeof(uchar));

//                 HyperspectralPanoramaChannelPtr channelPtr(new HyperspectralPanoramaChannel);
//                 channelPtr->channel = img;
//                 channelPtr->timestamp = timestamps[i];
//                 panoramaPtr->channels.push_back(channelPtr);
//             }
//             ret->panoramas.push_back(panoramaPtr);
//         }
//     }

//     return ret;
// }

/*template <typename Derived>
bool HyperspectralCameraIO<Derived>::isHyperspectralCamera(HighFive::Group& group)
{
    std::string id(HyperspectralCameraIO<Derived>::ID);
    std::string obj(HyperspectralCameraIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}*/

} // namespace lvr2

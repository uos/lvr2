#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

template <typename Derived>
void HyperspectralCameraIO<Derived>::saveHyperspectralCamera(
    const size_t& scanPosNo, const HyperspectralCameraPtr &buffer)
{
    Description d = m_featureBase->m_description->position(scanPosNo);
    
    // Setup default string for scan position
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    std::string groupName = sstr.str();

    // Override if descriptions contains a position string
    if(d.groupName)
    {
        groupName = *d.groupName;
    }
    saveHyperspectralCamera(groupName, buffer);
}

template <typename Derived>
void HyperspectralCameraIO<Derived>::saveHyperspectralCamera(
    std::string& group,
    const HyperspectralCameraPtr& hyperspectralCameraPtr)
{
    std::string id(HyperspectralCameraIO<Derived>::ID);
    std::string obj(HyperspectralCameraIO<Derived>::OBJID);
  
    //  hdf5util::setAttribute(group, "IO", id);
    //  hdf5util::setAttribute(group, "CLASS", obj);

    // saving estimated and registered pose
    m_matrixIO->saveMatrix(group, "extrinsics", hyperspectralCameraPtr->extrinsics);
    m_matrixIO->saveMatrix(group, "extrinsicsEstimate", hyperspectralCameraPtr->extrinsicsEstimate);

    std::vector<size_t> dim = {1, 1};
    std::vector<hsize_t> chunks = {1, 1};

    // saving focalLength
    doubleArr focalLength(new double[1]);
    focalLength[0] = hyperspectralCameraPtr->focalLength;
    m_arrayIO->saveDoubleArray(group, "focalLength", dim, focalLength);

    // saving offsetAngle
    doubleArr offsetAngle(new double[1]);
    offsetAngle[0] = hyperspectralCameraPtr->offsetAngle;
    m_arrayIO->saveDoubleArray(group, "offsetAngle", dim, offsetAngle);

    dim = {3, 1};
    chunks = {3, 1};

    doubleArr principal(new double[3]);
    principal[0] = hyperspectralCameraPtr->principal[0];
    principal[1] = hyperspectralCameraPtr->principal[1];
    principal[2] = hyperspectralCameraPtr->principal[2];
    m_arrayIO->saveDoubleArray(group, "principal", dim, principal);

    doubleArr distortion(new double[3]);
    distortion[0] = hyperspectralCameraPtr->distortion[0];
    distortion[1] = hyperspectralCameraPtr->distortion[1];
    distortion[2] = hyperspectralCameraPtr->distortion[2];
    m_arrayIO->saveDoubleArray(group, "distortion", dim, distortion);

    for (int i = 0; i < hyperspectralCameraPtr->panoramas.size(); i++)
    {
        HyperspectralPanoramaPtr panoramaPtr = hyperspectralCameraPtr->panoramas[i];

        ucharArr data(new unsigned char[hyperspectralCameraPtr->panoramas[i]->channels.size() *
                                        panoramaPtr->channels[0]->channel.rows *
                                        panoramaPtr->channels[0]->channel.cols]);

        std::memcpy(data.get(),
                    panoramaPtr->channels[0]->channel.data,
                    panoramaPtr->channels[0]->channel.rows *
                        panoramaPtr->channels[0]->channel.cols * sizeof(unsigned char));

        std::vector<size_t> dim = {hyperspectralCameraPtr->panoramas[i]->channels.size(),
                                   static_cast<size_t>(panoramaPtr->channels[0]->channel.rows),
                                   static_cast<size_t>(panoramaPtr->channels[0]->channel.cols)};

        for (int j = 1; j < hyperspectralCameraPtr->panoramas[i]->channels.size(); j++)
        {
            std::memcpy(data.get() + j * panoramaPtr->channels[j]->channel.rows *
                                         panoramaPtr->channels[j]->channel.cols,
                        panoramaPtr->channels[j]->channel.data,
                        panoramaPtr->channels[j]->channel.rows *
                            panoramaPtr->channels[j]->channel.cols * sizeof(unsigned char));
        }

        // generate group of panorama
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%08d", i);

        boost::filesystem::path nrpath(buffer);

        std::string panoramaGroup = (boost::filesystem::path(group) / nrpath).string();
        //HighFive::Group panoramaGroup = hdf5util::getGroup(group, nr_str);


        // save panorama
        m_arrayIO->saveUCharArray(panoramaGroup, "frames", dim, data);

        // save timestamps
        doubleArr timestamps(new double[hyperspectralCameraPtr->panoramas[i]->channels.size()]);
        size_t pos = 0;
        for (auto channel : hyperspectralCameraPtr->panoramas[i]->channels)
        {
            timestamps[pos++] = channel->timestamp;
        }
        dim = {pos, 1, 1};
        m_arrayIO->saveDoubleArray(panoramaGroup, "timestamps", dim, timestamps);
    }
}

template <typename Derived>
HyperspectralCameraPtr HyperspectralCameraIO<Derived>::loadHyperspectralCamera(const size_t& scanPosNo)
{
    HyperspectralCameraPtr ret(new HyperspectralCamera);
    
    Description d = m_featureBase->m_description->hyperspectralCamera(scanPosNo);
    
    // Default path
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    std::string groupName = "raw/" + sstr.str() + "/spectral/data";

    boost::optional<lvr2::Extrinsicsd> extrinsics = boost::none;
    boost::optional<lvr2::Extrinsicsd> extrinsicsEstimate = boost::none;
    boost::optional<double> focalLength = boost::none;
    boost::optional<double> offsetAngle = boost::none;
    boost::optional<lvr2::Vector3d> principal = boost::none;
    boost::optional<lvr2::Vector3d> distortion = boost::none;
   
    // Override defaults with scan project structure
    // definitions if valid
    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    // Try to load meta data from project description
    if(d.metaData)
    {
        if((*d.metaData)["extrinsics"])
        {
            try
            {
                // Load extrinsics from meta data 
                extrinsics = (*d.metaData)["extrinsics"].as<lvr2::Extrinsicsd>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }

        if((*d.metaData)["extrinsicsEstimate"])
        {
            try
            {
                // Load extrinsics from meta data 
                extrinsicsEstimate = (*d.metaData)["extrinsicsEstimate"].as<lvr2::Extrinsicsd>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }

        if((*d.metaData)["focalLength"])
        {
            try
            {
                focalLength = (*d.metaData)["focalLength"].as<double>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }

        if((*d.metaData)["offsetAngle"])
        {
            try
            {
                offsetAngle = (*d.metaData)["offsetAngle"].as<double>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }

        if((*d.metaData)["principal"])
        {
            try
            {
                principal = (*d.metaData)["principal"].as<Vector3d>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }

        if((*d.metaData)["distortion"])
        {
            try
            {
                distortion = (*d.metaData)["distortion"].as<Vector3d>();
            }
            catch(const YAML::BadConversion& e)
            {
                std::cout << timestamp << e.what() << std::endl;
            }
        }
    }

    // Check if meta data fields were loaded correctly. If not
    // try to load them via current file access kernel.
    if (!extrinsics)
    {
        Extrinsicsd e = m_matrixIO->template loadMatrix<Extrinsicsd>(groupName, "extrinsics");
        extrinsics = e;
    }
    ret->extrinsics = *extrinsics;
    

    // read extrinsicsEstimate
    if(!extrinsicsEstimate)
    {
        Extrinsicsd e = m_matrixIO-> template loadMatrix<Extrinsicsd>(groupName, "extrinsicsEstimate");
        extrinsicsEstimate = e;
    }
    ret->extrinsicsEstimate = *extrinsicsEstimate;
    

    if(!focalLength)
    {
        std::vector<size_t> dimension = {1};
        doubleArr focalLength = m_arrayIO->loadDoubleArray(groupName, "focalLength", dimension);

        if (dimension.size() != 1)
        {
            std::cout << timestamp << "HyperspectralCameraIO: Wrong focalLength dimension: " << dimension.size() << std::endl;
            std::cout << timestamp << "FocalLength will not be loaded." << std::endl;
        }
        else
        {
            ret->focalLength = focalLength[0];
        }
    }
    else
    {
        ret->focalLength = focalLength.get();
    }
    
       
    if(!offsetAngle)
    {
        std::vector<size_t> dimension = {1};
        doubleArr offsetAngle = m_arrayIO->loadDoubleArray(groupName, "offsetAngle", dimension);

        if (dimension.size() != 1)
        {
            std::cout << timestamp << "HyperspectralCameraIO: Wrong offsetAngle dimension: " << dimension.size() << std::endl;
            std::cout << timestamp << "OffsetAngle will not be loaded." << std::endl;
        }
        else
        {
            ret->offsetAngle = offsetAngle[0];
        }
    }
    else
    {
        ret->offsetAngle = offsetAngle.get();
    }
    

    if (!principal)
    {   
        std::vector<size_t> dimension = {3, 1};
        doubleArr principal = m_arrayIO->loadDoubleArray(groupName, "principal", dimension);

        if (dimension.size() != 3)
        {
            std::cout << timestamp << "HyperspectralCameraIO: Wrong principal dimension: " << dimension.size() << std::endl;
            std::cout << timestamp << "Principal point will not be loaded." << std::endl;
        }
        else
        {
            Vector3d p = {principal[0], principal[1], principal[2]};
            ret->principal = p;
        }
    }
    else
    {
        ret->principal = principal.get();
    }

    if (!distortion)
    {
        std::vector<size_t> dimension = {3, 1};
        doubleArr distortion = m_arrayIO->loadDoubleArray(groupName, "distortion", dimension);

        if (dimension.size() != 3)
        {
            std::cout << timestamp << "HyperspectralCameraIO: Wrong distortion dimension: " << dimension.size() << std::endl;
            std::cout << timestamp << "Distortion point will not be loaded." << std::endl;
        }
        else
        {
            Vector3d d = {distortion[0], distortion[1], distortion[2]};
            ret->distortion = d;
        }
    }
    else
    {
        ret->distortion = distortion.get();
    }
    
    // Iterate over all panoramas
    std::vector<std::string> positionGroups;
    m_featureBase->m_kernel->subGroupNames(groupName, std::regex("\\d{8}"), positionGroups);

    for (std::string positionGroup : positionGroups)
    {
        Description td = m_featureBase->m_description->hyperSpectralTimestamps(positionGroup);
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

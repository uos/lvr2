#include <boost/regex.hpp>
#include <opencv2/opencv.hpp>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/io/yaml/Scan.hpp"
#include "lvr2/io/yaml/ScanCamera.hpp"
#include "lvr2/io/yaml/ScanImage.hpp"

#include <sstream>

namespace lvr2
{

std::string getSensorType(const boost::filesystem::path& dir)
{
    std::string sensorType = "";

    // Try to open meta.yaml
    boost::filesystem::path metaPath = dir / "meta.yaml";
    if(boost::filesystem::exists(metaPath))
    {
        YAML::Node meta = YAML::LoadFile(metaPath.string());

        // Get sensor type 
        if(meta["sensor_type"])
        {
            sensorType = meta["sensor_type"].as<std::string>();
        }   
    }
    return sensorType;
}

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const size_t& positionNumber,
    const size_t& camNumber,
    const size_t& imageNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream camStr; 
    camStr << std::setfill('0') << std::setw(8) << camNumber;

    saveScanImage(root, image, posStr.str(), camStr.str(), imageNumber);
}

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const std::string positionDirectory,
    const size_t& cameraNumber,
    const size_t& imageNumber)
{
    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    saveScanImage(root, image, positionDirectory, camStr.str(), imageNumber);
}

boost::filesystem::path getScanImageDirectory(
    boost::filesystem::path root,
    const std::string positionDirectory,
    const std::string cameraDirectory)
{
    boost::filesystem::path pos(positionDirectory);
    boost::filesystem::path cam(cameraDirectory);
    boost::filesystem::path data("data/");
    return root / pos /cam / data;
}

void saveScanImage(
    const boost::filesystem::path& root,
    const ScanImage& image,
    const std::string positionDirectory,
    const std::string cameraDirectory,
    const size_t& imageNumber)
{
    // Some directory parsing stuff
    std::stringstream metaFileName;
    metaFileName << std::setfill('0') << std::setw(8) << imageNumber << ".yaml";

    std::stringstream imageFileName;
    imageFileName << std::setfill('0') << std::setw(8) << imageNumber << ".png";

    boost::filesystem::path imageDirectory = 
        getScanImageDirectory(root, positionDirectory, cameraDirectory);

    boost::filesystem::path imagePath(imageDirectory / boost::filesystem::path(metaFileName.str()));
    boost::filesystem::path metaPath(imageDirectory / boost::filesystem::path(metaFileName.str()));

    // Write meta information for scan image
    YAML::Node meta;
    meta = image;

    std::ofstream out(metaPath.c_str());
    if (out.good())
    {
        std::cout << timestamp << "Writing " << metaPath << std::endl;
        out << meta;
    }
    else
    {
        std::cout << timestamp 
            << "Warning: to write " << metaPath << std::endl;     
    }

    // Write image data
    std::cout << timestamp << "Writing " << imagePath << std::endl;
    cv::imwrite(imagePath.string(), image.image);
}

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const size_t& positionNumber,
    const size_t& cameraNumber,
    const size_t& imageNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    return loadScanImage(root, image, posStr.str(), camStr.str(), imageNumber);
}

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const std::string& positionDirectory,
    const size_t& cameraNumber,
    const size_t& imageNumber)
{
    std::stringstream camStr; 
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;
    
    return loadScanImage(root, image, positionDirectory, camStr.str(), imageNumber);
}

bool loadScanImage(
    const boost::filesystem::path& root,
    ScanImage& image,
    const std::string& positionDirectory,
    const std::string& cameraDirectory,
    const size_t& imageNumber)
{
    // Some directory parsing stuff
    std::stringstream metaFileName;
    metaFileName << std::setfill('0') << std::setw(8) << imageNumber << ".yaml";

    std::stringstream imageFileName;
    imageFileName << std::setfill('0') << std::setw(8) << imageNumber << ".png";

    boost::filesystem::path imageDirectory = 
        getScanImageDirectory(root, positionDirectory, cameraDirectory);

    boost::filesystem::path imagePath(imageDirectory / boost::filesystem::path(metaFileName.str()));
    boost::filesystem::path metaPath(imageDirectory / boost::filesystem::path(metaFileName.str()));

    // Load meta info
    std::cout << timestamp << "Loading " << metaPath << std::endl;
    YAML::Node meta = YAML::LoadFile(metaPath.string());
    image = meta.as<ScanImage>();

    // Load image data
    std::cout << timestamp << "Loading " << imagePath << std::endl;
    image.imageFile = imagePath;
    image.image = cv::imread(imagePath.string());

    return true;
}

void loadScanImages(
    vector<ScanImagePtr>& images, 
    boost::filesystem::path dataPath)
{
    bool stop = false;
    size_t c = 1;
    while(!stop)
    {
        stringstream metaStr;
        metaStr << std::setfill('0') << std::setw(8) << c << ".yaml";

        stringstream pngStr;
        metaStr << std::setfill('0') << std::setw(8) << c << ".png";

        boost::filesystem::path metaPath = dataPath / metaStr.str();
        boost::filesystem::path pngPath = dataPath / pngStr.str();

        // Check if both .png and corresponding .yaml file exist
        if(boost::filesystem::exists(metaPath) && boost::filesystem::exists(pngPath) && getSensorType(dataPath) == "ScanImage")
        {
            // Load meta info
            ScanImage* image = new ScanImage;
            
            std::cout << timestamp << "Loading " << metaPath << std::endl;
            YAML::Node meta = YAML::LoadFile(metaPath.string());
            *image = meta.as<ScanImage>();

            // Load image data
             std::cout << timestamp << "Loading " << pngPath << std::endl;
            image->imageFile = pngPath;
            image->image = cv::imread(pngPath.string());

            // Store new image
            images.push_back(ScanImagePtr(image));
            c++;
        }
        else
        {
            std::cout 
                << timestamp << "Read " << c << " images from " << dataPath << std::endl;        
            stop = true;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// SCANCAMERA
///////////////////////////////////////////////////////////////////////////////////////

boost::filesystem::path getScanCameraDirectory(
    boost::filesystem::path root,
    const std::string positionDirectory,
    const std::string cameraDirectory)
{
    boost::filesystem::path pos(positionDirectory);
    boost::filesystem::path cam(cameraDirectory);
    return root / pos /cam;
}


void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& camera,
    const std::string positionDirectory,
    const std::string cameraDirectory)
{
    boost::filesystem::path cameraPath = 
        getScanCameraDirectory(root, positionDirectory, cameraDirectory);


    boost::filesystem::path metaPath(cameraPath / "meta.yaml");

     // Write meta information of scan camera
    YAML::Node meta;
    meta = camera;

    std::ofstream out(metaPath.c_str());
    if (out.good())
    {
        std::cout << timestamp << "Writing " << metaPath << std::endl;
        out << meta;
    }
    else
    {
        std::cout << timestamp 
            << "Warning: Unable to write " << metaPath << std::endl;     
    }

    // Save all images
    for(size_t i = 0; i < camera.images.size(); i++)
    {
        saveScanImage(root, *camera.images[i], positionDirectory, cameraDirectory, i);
    }
}

void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& camera,
    const size_t& positionNumber,
    const size_t& cameraNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    return saveScanCamera(root, camera, posStr.str(), camStr.str());
}

void saveScanCamera(
    const boost::filesystem::path& root,
    const ScanCamera& camera,
    const std::string& positionDirectory,
    const size_t& cameraNumber)
{
    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    return saveScanCamera(root, camera, positionDirectory, camStr.str());
}

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& camera,
    const std::string& positionDirectory,
    const size_t& cameraNumber)
{
    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    return loadScanCamera(root, camera, positionDirectory, camStr.str());
}

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& camera,
    const size_t& positionNumber,
    const size_t& cameraNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream camStr;
    camStr << std::setfill('0') << std::setw(8) << cameraNumber;

    return loadScanCamera(root, camera, posStr.str(), camStr.str());
}

bool loadScanCamera(
    const boost::filesystem::path& root,
    ScanCamera& camera,
    const std::string& positionDirectory,
    const std::string& cameraDirectory)
{
    boost::filesystem::path cameraPath = 
        getScanCameraDirectory(root, positionDirectory, cameraDirectory);

    if(getSensorType(cameraPath) == "ScanCamera")
    {

        boost::filesystem::path metaPath = cameraPath / "meta.yaml";

        // Load camera data
        std::cout << timestamp << "Loading " << metaPath << std::endl;
        YAML::Node meta = YAML::LoadFile(metaPath.string());
        camera = meta.as<ScanCamera>();

        // Load all scan images
        loadScanImages(camera.images, cameraPath / "data");
    }
    else
    {
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// SCAN
///////////////////////////////////////////////////////////////////////////////////////

void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const std::string positionDirectory,
    const std::string scanDirectory,
    const std::string scanName)
{
    boost::filesystem::path positionDirectoryPath(positionDirectory);
    boost::filesystem::path scanDirectoryPath(scanDirectory);
    boost::filesystem::path scanNamePath(scanName);

    // Check if meta.yaml exists for given directory
    boost::filesystem::path metaPath = root / positionDirectoryPath / scanDirectoryPath / "meta.yaml";
    if(!boost::filesystem::exists(metaPath))
    {
        YAML::Node node;
        node["sensor_type"] = "Scan";
        node["sensor_name"] = "Scanner";

        std::ofstream out(metaPath.c_str());
        if (out.good())
        {
            std::cout << timestamp << "Writing " << metaPath << std::endl;
            out << node;
        }
        else
        {
            std::cout << timestamp
                      << "Warning: Unable to write " << metaPath << std::endl;
        }
    }

    boost::filesystem::path outputDir = root / positionDirectoryPath / scanDirectoryPath / "data";
    boost::filesystem::path scanOut = outputDir / scanNamePath / ".ply";
    boost::filesystem::path scanMetaOut = outputDir / scanNamePath / ".yaml";

    // Write meta.yaml
    YAML::Node node;
    node = scan;

    std::ofstream out(scanMetaOut.string());
    if (out.good())
    {
        std::cout << timestamp << "Writing " << scanMetaOut << std::endl;
        out << node;
    }
    else
    {
        std::cout << timestamp
                  << "Warning: Unable to write " << scanMetaOut << std::endl;
    }

    // Write point cloud data
    ModelPtr model(new Model);
    model->m_pointCloud = scan.points;
    ModelFactory::saveModel(model, scanOut.string());
}

void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const std::string positionDirectory,
    const std::string scanDirectory,
    const size_t& scanNumber)
{
    std::stringstream scanStr;
    scanStr << std::setfill('0') << std::setw(8) << scanNumber;

    saveScan(root, scan, positionDirectory, scanDirectory, scanStr.str());
}

void saveScan(
    const boost::filesystem::path& root,
    const Scan& scan,
    const size_t& positionNumber,
    const size_t& scanNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream scanStr;
    scanStr << std::setfill('0') << std::setw(8) << scanNumber << ".ply";

    saveScan(root, scan, posStr.str(), "scans", scanStr.str());
}

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const std::string& positionDirectory,
    const std::string& scanSubDirectory,
    const std::string& scanName)
{
    // Convert strings to paths
    boost::filesystem::path positionPath;
    boost::filesystem::path scanSubDirectoryPath;
    boost::filesystem::path scanNamePath;
    
    boost::filesystem::path scanDirectoryPath = root / scanSubDirectoryPath;
    boost::filesystem::path scanDataPath = scanSubDirectoryPath / "data";
    if(getSensorType(scanDirectoryPath) == "Scan")
    {
        // Load meta data
        boost::filesystem::path metaPath = scanDataPath / (scanName + ".yaml");
        std::cout << timestamp << "Loading " << metaPath << std::endl;
        YAML::Node meta = YAML::LoadFile(metaPath.string());
        scan = meta.as<Scan>();

        // Load scan
        boost::filesystem::path scanFile = scanDataPath / (scanName + ".ply");
        std::cout << timestamp << "Loading " << scanFile << std::endl;
        ModelPtr model = ModelFactory::readModel(scanFile.string());

        if(model->m_pointCloud)
        {
            scan.points = model->m_pointCloud;
        }
        else
        {
            std::cout << timestamp 
                      << "Warning: Loading " << scanFile << " failed." << std::endl;
            return false;
        }
        

        return true;
    }
    
    return false;
}

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const std::string& positionDirectory,
    const size_t& scanNumber)
{
    std::stringstream scanStr;
    scanStr << std::setfill('0') << std::setw(8) << scanNumber;

    return loadScan(root, scan, positionDirectory, "scans", scanStr.str());
}

bool loadScan(
    const boost::filesystem::path& root,
    Scan& scan,
    const size_t& positionNumber,
    const size_t& scanNumber)
{
    std::stringstream posStr;
    posStr << std::setfill('0') << std::setw(8) << positionNumber;

    std::stringstream scanStr;
    scanStr << std::setfill('0') << std::setw(8) << scanNumber;

    return loadScan(root, scan, posStr.str(), "scans", scanStr.str());
}

///////////////////////////////////////////////////////////////////////////////////////
/// SCAN_POSITION
///////////////////////////////////////////////////////////////////////////////////////
void saveScanPosition(
    const boost::filesystem::path& root,
    const ScanPosition& scanPos,
    const std::string positionDirectory)
{

}

void saveScanPosition(
    const boost::filesystem::path& root,
    const ScanPosition& scanPos,
    const size_t& positionNumber)
{

}

bool loadScanPosition(
    const boost::filesystem::path& root,
    ScanPosition& scanPos,
    const std::string& positionDirectory)
{

}

bool loadScanPosition(
    const boost::filesystem::path& root,
    ScanPosition& scanPos,
    const size_t& positionNumber)
{
    
}





























// std::set<size_t> loadPositionIdsFromDirectory(
//     const boost::filesystem::path& path
// )
// {
//     std::set<size_t> positions;
//     boost::filesystem::path scan_path = path / "scans";
//     const boost::regex scan_dir_filter("^([0-9]{5}).*$");

//     boost::filesystem::directory_iterator end_itr;
//     for(boost::filesystem::directory_iterator it(scan_path); it != end_itr; ++it )
//     {
//         if( !boost::filesystem::is_directory( it->path() ) ) continue;
//         boost::smatch what;
//         if( !boost::regex_match( it->path().filename().string(), what, scan_dir_filter ) ) continue;
//         positions.insert(static_cast<size_t>(std::stoul(what[1])));
//     }

//     return positions;
// }

// std::set<size_t> loadCamIdsFromDirectory(
//     const boost::filesystem::path& path,
//     const size_t& positionNr
// )
// {
//     std::set<size_t> cam_ids;

//     std::stringstream ss;
//     ss << std::setfill('0') << std::setw(5) << positionNr;
//     boost::filesystem::path scan_images_dir = path / "scan_images" / ss.str();

//     if(boost::filesystem::exists(scan_images_dir))
//     {
//         const boost::regex image_filter("^(\\d+)_(\\d+)\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras|tiff|tif)$");

//         boost::filesystem::directory_iterator end_itr;
//         for(boost::filesystem::directory_iterator it(scan_images_dir); it != end_itr; ++it )
//         {
//             if( !boost::filesystem::is_regular_file( it->path() ) ) continue;

//             boost::smatch what;
//             if( !boost::regex_match( it->path().filename().string(), what, image_filter ) ) continue;

//             cam_ids.insert(static_cast<size_t>(std::stoul(what[1])));

//         }
//     }

//     return cam_ids;
// }

// std::set<size_t> loadImageIdsFromDirectory(
//     const boost::filesystem::path& path,
//     const size_t& positionNr,
//     const size_t& camNr
// )
// {
//     std::set<size_t> img_ids;

//     std::stringstream ss;
//     ss << std::setfill('0') << std::setw(5) << positionNr;
//     boost::filesystem::path scan_images_dir = path / "scan_images" / ss.str();

    
//     if(boost::filesystem::exists(scan_images_dir))
//     {
//         const boost::regex image_filter("^(\\d+)_(\\d+)\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras|tiff|tif)$");

//         boost::filesystem::directory_iterator end_itr;
//         for(boost::filesystem::directory_iterator it(scan_images_dir); it != end_itr; ++it )
//         {
//             if( !boost::filesystem::is_regular_file( it->path() ) ) continue;

//             boost::smatch what;
//             if( !boost::regex_match( it->path().filename().string(), what, image_filter ) ) continue;

//             if(static_cast<size_t>(std::stoul(what[1])) == camNr)
//             {
//                 img_ids.insert(static_cast<size_t>(std::stoul(what[2])));
//             }
//         }
//     }


//     return img_ids;
// }

// void writeScanMetaYAML(const boost::filesystem::path& path, const Scan& scan)
// {
    
//     YAML::Node meta;
//     meta = scan;

//     std::ofstream out(path.c_str());
//     if (out.good())
//     {
//         out << meta;
//     }
//     else
//     {
//         std::cout << timestamp << "Warning: Unable to open " 
//                   << path.string() << "for writing." << std::endl;
//     }
    
// }

// void saveScanToDirectory(
//     const boost::filesystem::path& path, 
//     const Scan& scan, 
//     const size_t& positionNr)
// {
//     // Create full path from root and scan number
//     std::stringstream ss;
//     ss << std::setfill('0') << std::setw(5) << positionNr;
//     boost::filesystem::path directory = path / "scans" / ss.str();
    
//     // Check if directory exists, if not create it and write meta data
//     // and into the new directory

//     if(!boost::filesystem::exists(path))
//     {
//         std::cout << timestamp << "Creating " << path << std::endl;
//         boost::filesystem::create_directory(path);
//     }

//     if(!boost::filesystem::exists(path / "scans")) {
//         std::cout << timestamp << "Creating " << path / "scans" << std::endl;
//         boost::filesystem::create_directory(path / "scans");
//     }

//     if(!boost::filesystem::exists(directory)) 
//     {
//         std::cout << timestamp << "Creating " << directory << std::endl;
//         boost::filesystem::create_directory(directory);
//     } 

//     // Create yaml file with meta information
//     std::cout << timestamp << "Creating scan meta.yaml..." << std::endl;
//     writeScanMetaYAML(directory / "meta.yaml", scan);

//     // Save points
//     std::string scanFileName( (directory / "scan.ply").string() );
//     std::cout << timestamp << "Writing " << scanFileName << std::endl;
    
//     ModelPtr model(new Model());
//     if(scan.m_points)
//     {
//         model->m_pointCloud = scan.m_points;
//         PLYIO io;
//         io.save(model, scanFileName);
//     }
//     else
//     {
//         std::cout << timestamp << "Warning: Point cloud pointer is empty!" << std::endl;
//     }
   
    
// }

// bool loadScanFromDirectory(
//     const boost::filesystem::path& path, 
//     Scan& scan, const size_t& positionNr, bool load)
// {
//     if(boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
//     {
//         std::stringstream ss;
//         ss << std::setfill('0') << std::setw(5) << positionNr;
//         boost::filesystem::path position_directory = path / "scans" / ss.str();
        
//         if(boost::filesystem::exists(position_directory))
//         {
//             // Load meta data
//             boost::filesystem::path meta_yaml_path = position_directory / "meta.yaml";
//             std::cout << timestamp << "Reading " << meta_yaml_path << std::endl;
//             loadScanMetaInfoFromYAML(meta_yaml_path, scan);


//             // Load scan data
//             boost::filesystem::path scan_path = position_directory / "scan.ply";
//             std::cout << timestamp << "Reading " << scan_path << std::endl;

//             if (boost::filesystem::exists(scan_path))
//             {
//                 scan.m_scanFile = scan_path;
//                 scan.m_scanRoot = path; // TODO: Check root dir or scan dir??
//                 if (load)
//                 {
//                     PLYIO io;
//                     ModelPtr model = io.read(scan_path.string());
//                     scan.m_points = model->m_pointCloud;
//                     scan.m_pointsLoaded = true;
//                 }
//                 else
//                 {
//                     scan.m_pointsLoaded = false;
//                 }
//                 return true;
//             }
//             else
//             {
//                 std::cout << "Warning: scan.ply not found in directory "
//                           << scan_path << std::endl;
//                 return false;
//             }
//         }
//         else
//         {
//             std::cout << timestamp
//                       << "Warning: Scan directory " << position_directory << " "
//                       << "does not exist." << std::endl;
//             return false;
//         }
//     }
//     else
//     {
//         std::cout << timestamp 
//                   << "Warning: '" << path.string() 
//                   << "' does not exist or is not a directory." << std::endl; 
//         return false;
//     }
// }

// void loadScanMetaInfoFromYAML(const boost::filesystem::path& path, Scan& scan)
// {
//     YAML::Node meta = YAML::LoadFile(path.string());
//     scan = meta.as<Scan>();
// }


// void saveScanToHDF5(const std::string filename, const size_t& positionNr)
// {

// }

// bool loadScanFromHDF5(const std::string filename, const size_t& positionNr)
// {
//     return true;
// }

// void saveScanImageToDirectory(
//     const boost::filesystem::path& path, 
//     const std::string& camDir,
//     const ScanImage& image,
//     const size_t& positionNr,
//     const size_t& imageNr)
// {
//     std::stringstream pos_ss;
//     pos_ss << std::setfill('0') << std::setw(5) << positionNr;
//     boost::filesystem::path scanimage_directory = path / pos_ss.str() / camDir;

//     // Create directory for scan image if necessary
//     if(!boost::filesystem::exists(scanimage_directory))
//     {
//         boost::filesystem::create_directory(scanimage_directory);
//     }

//     // Create directory for scan image data frame data if necessary
//     boost::filesystem::path scanimage_data_directory = scanimage_directory / "data";
//     if(!boost::filesystem::exists(scanimage_data_directory))
//     {
//         boost::filesystem::create_directory(scanimage_data_directory);
//     }

    
    
// }

// void writePinholeModelToYAML(
//     const boost::filesystem::path& path, const PinholeModeld& model)
// {
//     // YAML::Node meta;
//     // meta = model;
    
//     // std::ofstream out(path.c_str());
//     // if (out.good())
//     // {
//     //     out << meta;
//     // }
//     // else
//     // {
//     //     std::cout << timestamp << "Warning: Unable to open " 
//     //               << path.string() << "for writing." << std::endl;
//     // }

// }

// void loadPinholeModelFromYAML(const boost::filesystem::path& path, PinholeModeld& model)
// {
// //     YAML::Node model_file = YAML::LoadFile(path.string());
// //     model = model_file.as<PinholeModeld>();
// // 
// }

// bool loadScanImageFromDirectory(
//     const boost::filesystem::path& path, 
//     ScanImage& image, 
//     const size_t& positionNr, 
//     const size_t& camNr,
//     const size_t& imageNr)
// {
//     // // Convert position and image number to strings
//     // stringstream pos_str;
//     // pos_str << std::setfill('0') << std::setw(5) << positionNr;

//     // // Construct a path to image directory and check
//     // boost::filesystem::path scan_image_dir = path / "scan_images" / pos_str.str();
//     // if(boost::filesystem::exists(scan_image_dir))
//     // {
//     //     std::stringstream yaml_file, image_file;
//     //     yaml_file << camNr << "_" << imageNr << ".yaml";
//     //     image_file << camNr << "_" << imageNr << ".png";

//     //     boost::filesystem::path meta_path = scan_image_dir / yaml_file.str();
//     //     boost::filesystem::path image_path = scan_image_dir / image_file.str();

//     //     if(!boost::filesystem::exists(meta_path))
//     //     {
//     //         std::cout << timestamp << "Could not load meta file of scan/cam/img: " << positionNr << "/" << camNr << "/" << imageNr << std::endl;
//     //         return false;
//     //     }

//     //     std::cout << timestamp << "Loading " << image_path << std::endl;
//     //     image.image = cv::imread(image_path.string(), 1);
//     //     image.imageFile = image_path;

//     //     std::cout << timestamp << "Loading " << meta_path << std::endl;
//     //     loadPinholeModelFromYAML(meta_path, image.camera);
//     // }
//     // else
//     // {
//     //     std::cout << timestamp << "Warning: Image directory does not exist: "
//     //               << scan_image_dir << std::endl;
//     //     return false;
//     // }

//     return true;
// }

// void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr)
// {  
//     // // Save scan data
//     // std::cout << timestamp << "Saving scan postion " << positionNr << std::endl;
//     // if(position.scan)
//     // {
//     //     saveScanToDirectory(path, *position.scan, positionNr);
//     // }
//     // else
//     // {
//     //     std::cout << timestamp << "Warning: Scan position " << positionNr
//     //               << " contains no scan data." << std::endl;
//     // }
    
//     // // Save rgb camera recordings
//     // for(size_t cam_id = 0; cam_id < position.cams.size(); cam_id++)
//     // {
//     //     // store each image of camera
//     //     for(size_t img_id = 0; img_id < position.cams[cam_id]->images.size(); img_id++ )
//     //     {
//     //         saveScanImageToDirectory(path, *position.cams[cam_id]->images[img_id], positionNr, cam_id, img_id);
//     //     }
//     // }
// }

// void get_all(
//     const boost::filesystem::path& root, 
//     const string& ext, vector<boost::filesystem::path>& ret)
// {
//     if(!boost::filesystem::exists(root) || !boost::filesystem::is_directory(root))
//     {
//         return;
//     } 

//     boost::filesystem::directory_iterator it(root);
//     boost::filesystem::directory_iterator endit;

//     while(it != endit)
//     {
//         if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ext)
//         {
//             ret.push_back(it->path().filename());
//         } 
//         ++it;
//     }
// }

// bool loadScanPositionFromDirectory(
//     const boost::filesystem::path& path,
//     ScanPosition& position, 
//     const size_t& positionNr)
// {
//     // bool scan_read = false;
//     // bool images_read = false;

//     // std::cout << timestamp << "Loading scan position " << positionNr << std::endl;
//     // Scan scan;
//     // if(!loadScanFromDirectory(path, scan, positionNr, true))
//     // {
//     //     std::cout << timestamp << "Warning: Scan position " << positionNr 
//     //               << " does not contain scan data." << std::endl;
//     // } else {
//     //     position.scan = scan;
//     // }


//     // boost::filesystem::path img_directory = path / "scan_images";
//     // if(boost::filesystem::exists(img_directory))
//     // {
//     //     std::stringstream ss;
//     //     ss << std::setfill('0') << std::setw(5) << positionNr;
//     //     boost::filesystem::path scanimage_directory = img_directory / ss.str();
//     //     if(boost::filesystem::exists(scanimage_directory))
//     //     {
//     //         std::set<size_t> cam_ids = loadCamIdsFromDirectory(path, positionNr);

//     //         for(const size_t& cam_id : cam_ids)
//     //         {
//     //             ScanCameraPtr cam(new ScanCamera);
//     //             std::set<size_t> img_ids = loadImageIdsFromDirectory(path, positionNr, cam_id);
//     //             for(const size_t& img_id : img_ids)
//     //             {
//     //                 ScanImagePtr img(new ScanImage);
//     //                 loadScanImageFromDirectory(path, *img, positionNr, cam_id, img_id);
//     //                 cam->images.push_back(img);
//     //             }
//     //             position.cams.push_back(cam);
//     //         }
//     //     } else {
//     //         std::cout << timestamp << "Warning: Specified scan has no images." << std::endl;
//     //     }

//     // }
//     // else
//     // {
//     //     std::cout << timestamp << "Scan position " << positionNr 
//     //               << " has no images." << std::endl;
//     // }
//     // return true;
// }

// void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& project)
// {
//     boost::filesystem::create_directory(path);

//     YAML::Node yaml;
//     yaml["pose"] = project.pose;
//     std::ofstream out( (path / "meta.yaml").string() );
//     if (out.good())
//     {
//         out << yaml;
//     }
//     else
//     {
//         std::cout << timestamp << "Warning: Unable to open " 
//                   << path.string() << "for writing." << std::endl;
//     }

//     for(size_t i = 0; i < project.positions.size(); i++)
//     {
//         saveScanPositionToDirectory(path, *project.positions[i], i);
//     }
// }

// bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& project)
// {
//     YAML::Node meta = YAML::LoadFile((path / "meta.yaml").string() );    
//     project.pose = meta["pose"].as<Transformd>();


//     boost::filesystem::directory_iterator it(path / "scans");
//     boost::filesystem::directory_iterator endit;

//     std::set<size_t> scan_pos_ids = loadPositionIdsFromDirectory(path);

//     for(size_t scan_pos_id : scan_pos_ids)
//     {
//         ScanPositionPtr scan_pos(new ScanPosition);
//         loadScanPositionFromDirectory(path, *scan_pos, scan_pos_id);
//         project.positions.push_back(scan_pos);
//     }
    

//     return true;
// }

} // namespace lvr2
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/hdf5/ScanIO.hpp"

#include <opencv2/opencv.hpp>
namespace lvr2
{
using ScanHDF5IO = lvr2::Hdf5Build<lvr2::hdf5features::ScanIO>;

void writeScanMetaYAML(const boost::filesystem::path& path, const Scan& scan)
{
    YAML::Node meta;

    // Write start and end time
    meta["start_time"]  = scan.m_startTime;
    meta["end_time"]    = scan.m_endTime;

    // Write pose estimation and registration

    // const double* poseData = scan.m_poseEstimation.data();
    // const double* registrationData = scan.m_registration.data();
    // for(size_t i = 0; i < 16; i++)
    // {
    //     meta["pose_estimate"].push_back(poseData[i]);
    //     meta["registration"].push_back(registrationData[i]);
    // }

    saveMatrixToYAML<double, 4, 4>(meta, std::string("pose_estimate"), scan.m_poseEstimation);
    saveMatrixToYAML<double, 4, 4>(meta, std::string("registration"), scan.m_registration);

    // Write scan configutaration parameters
    YAML::Node config;
    config["theta"].push_back(scan.m_thetaMin);
    config["theta"].push_back(scan.m_thetaMax);

    config["phi"].push_back(scan.m_phiMin);
    config["phi"].push_back(scan.m_phiMax);

    config["v_res"] = scan.m_vResolution;
    config["h_res"] = scan.m_hResolution;

    config["num_points"] = scan.m_numPoints;

    // Add configuration group
    meta["config"] = config;

    std::ofstream out(path.c_str());
    if (out.good())
    {
        out << meta;
    }
    else
    {
        std::cout << timestamp << "Warning: Unable to open " 
                  << path.string() << "for writing." << std::endl;
    }
    
}

void saveScanToDirectory(
    const boost::filesystem::path& path, 
    const Scan& scan, 
    const size_t& positionNr)
{
    // Create full path from root and scan number
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << positionNr;
    boost::filesystem::path directory = path / "scans" / ss.str();
    
    // Check if directory exists, if not create it and write meta data
    // and into the new directory
    if(!boost::filesystem::exists(path))
    {
        std::cout << timestamp << "Creating " << directory << std::endl;
        boost::filesystem::create_directory(directory);        

        // Create yaml file with meta information
        std::cout << timestamp << "Creating scan meta.yaml..." << std::endl;
        writeScanMetaYAML(directory, scan);

        // Save points
        std::string scanFileName(directory.string() + "scan.ply");
        std::cout << timestamp << "Writing " << scanFileName << std::endl;
        
        ModelPtr model(new Model());
        if(scan.m_points)
        {
            model->m_pointCloud = scan.m_points;
            PLYIO io;
            io.save(model, scanFileName);
        }
        else
        {
            std::cout << timestamp << "Warning: Point cloud pointer is empty!" << std::endl;
        }
    }
    else
    {
        std::cout << timestamp 
                  << "Warning: Directory " << path 
                  << " already exists. Will not override..." << std::endl;
    }
    
}

bool loadScanFromDirectory(
    const boost::filesystem::path& path, 
    Scan& scan, const size_t& positionNr, bool load)
{
    if(boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << positionNr;
        boost::filesystem::path position_directory = path / "scans" / ss.str();
        
        if(boost::filesystem::exists(position_directory))
        {
            // Load meta data
            boost::filesystem::path meta_yaml_path = position_directory / "meta.yaml";
            std::cout << timestamp << "Reading " << meta_yaml_path << std::endl;
            loadScanMetaInfoFromYAML(meta_yaml_path, scan);

            // Load scan data
            boost::filesystem::path scan_path = position_directory / "scan.ply";
            std::cout << timestamp << "Reading " << scan_path << std::endl;

            if (boost::filesystem::exists(scan_path))
            {
                scan.m_scanFile = scan_path;
                scan.m_scanRoot = path; // TODO: Check root dir or scan dir??
                if (load)
                {
                    PLYIO io;
                    ModelPtr model = io.read(scan_path.string());
                    scan.m_points = model->m_pointCloud;
                    scan.m_pointsLoaded = true;
                }
                else
                {
                    scan.m_pointsLoaded = false;
                }
                return true;
            }
            else
            {
                std::cout << "Warning: scan.ply not found in directory "
                          << scan_path << std::endl;
                return false;
            }
        }
        else
        {
            std::cout << timestamp
                      << "Warning: Scan directory " << position_directory << " "
                      << "does not exist." << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << timestamp 
                  << "Warning: '" << path.string() 
                  << "' does not exist or is not a directory." << std::endl; 
        return false;
    }
}

void loadScanMetaInfoFromYAML(const boost::filesystem::path& path, Scan& scan)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(path.string());

    for (auto &n : root)
    {
        for (YAML::const_iterator it = n.begin(); it != n.end(); ++it)
        {
             if (it->first.as<string>() == "start_time")
             {
                scan.m_startTime = it->second.as<float>();
             }
             else if (it->first.as<string>() == "end_time")
             {
                scan.m_endTime = it->second.as<float>();
             }
             else if (it->first.as<string>() == "pose_estimate")
             {
                scan.m_poseEstimation = loadMatrixFromYAML<double, 4, 4>(it);
             }
             else if (it->first.as<string>() == "registration")
             {
                scan.m_registration = loadMatrixFromYAML<double, 4, 4>(it);
             }
             else if (it->first.as<string>() == "config")
             {
                YAML::Node config = it->second;
                if(config["theta"])
                {
                    YAML::Node tmp = config["theta"];
                    scan.m_thetaMin = tmp[0].as<float>();
                    scan.m_thetaMax = tmp[1].as<float>();
                }
                
                if(config["phi"])
                {
                    YAML::Node tmp = config["phi"];
                    scan.m_phiMin = tmp[0].as<float>();
                    scan.m_phiMin = tmp[1].as<float>();
                }
                
                if(config["v_res"])
                {
                    scan.m_vResolution = config["v_res"].as<float>();
                }
                
                if(config["h_res"])
                {
                    scan.m_hResolution = config["h_res"].as<float>();
                }
                
                if(config["num_points"])
                {
                    scan.m_numPoints = config["num_points"].as<size_t>();
                } 
             }
        }
    }
}


void saveScanToHDF5(const std::string filename, Scan& scan, const size_t& positionNr)
{
    ScanHDF5IO scanIO;
    scanIO.open(filename);
    std::string groupName = "/raw/scans/position_";
    std::string number = std::to_string(positionNr);
    groupName = groupName + std::string(5-number.length(), '0') + number;
    std::cout << groupName << std::endl;
    scanIO.save(groupName, scan);
}

ScanPtr loadScanFromHDF5(const std::string filename, const size_t& positionNr)
{
    ScanHDF5IO scanIO;
    scanIO.open(filename);
    std::string groupName = "/raw/scans/position_";
    std::string number = std::to_string(positionNr);
    groupName = groupName + std::string(5-number.length(), '0') + number;

    // ScanPtr loadedScan = scanIO.loadScan(groupName);
    ScanPtr loadedScan = scanIO.loadPreview(groupName);
    if(loadedScan)
    {
        loadedScan->m_positionNumber = positionNr;
    }
    return loadedScan;
}

void saveScanImageToDirectory(
    const boost::filesystem::path& path, 
    const ScanImage& image, const size_t& positionNr, const size_t& imageNr)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << positionNr;
    boost::filesystem::path scanimage_directory = path / "scan_images" / ss.str();

    if(boost::filesystem::exists(path))
    {
        // Create scan image directory if necessary
        if(!boost::filesystem::exists(scanimage_directory))
        {
            std::cout << timestamp << "Creating " << scanimage_directory << std::endl;
            boost::filesystem::create_directory(scanimage_directory);        
        }
        

        // Create in image folder for current position if necessary
        std::stringstream position_str;
        position_str << std::setfill('0') << std::setw(5) << positionNr;
        boost::filesystem::path position_directory = scanimage_directory / position_str.str();

        if(!boost::filesystem::exists(position_directory))
        {
            std::cout << timestamp << "Creating " << position_directory << std::endl;
            boost::filesystem::create_directory(position_directory);
        }
        
        // Save image in .png format
        std::stringstream image_str;
        image_str << "img_" << std::setfill('0') << std::setw(3) << imageNr << ".png";
        boost::filesystem::path image_path = position_directory / image_str.str();
        
        std::cout << timestamp << "Saving " << image_path << std::endl;
        cv::imwrite(image_path.string(), image.image);

        // Save calibration yaml
        std::stringstream meta_str;
        meta_str << "img:" << std::setfill('0') << std::setw(3) << imageNr << ".yaml";
        boost::filesystem::path meta_path = position_directory / meta_str.str();
        std::cout << timestamp << "Saving " << meta_path << std::endl;
        writePinholeModelToYAML(meta_path, image.camera);
    }
    else
    {
        std::cout << timestamp
                  << "Warning: Scan directory for scan image does not exist: "
                  << path << std::endl;
    }
    
}

void writePinholeModelToYAML(
    const boost::filesystem::path& path, const PinholeCameraModeld& model)
{
    YAML::Node meta;
     
    
    Intrinsicsd intrinsics = model.intrinsics();
    Distortiond distortion = model.distortion();

    saveMatrixToYAML<double, 4, 4>(meta, std::string("extrinsic"), model.extrinsics());
    saveMatrixToYAML<double, 4, 4>(meta, std::string("extrinsic"), model.extrinsicsEstimate());
    saveMatrixToYAML<double, 3, 3>(meta, std::string("intrinsic"), model.intrinsics());
    saveMatrixToYAML<double, 6, 1>(meta, std::string("distortion"), model.distortion());

    std::ofstream out(path.c_str());
    if (out.good())
    {
        out << meta;
    }
    else
    {
        std::cout << timestamp << "Warning: Unable to open " 
                  << path.string() << "for writing." << std::endl;
    }

}

void loadPinholeModelFromYAML(const boost::filesystem::path& path, PinholeCameraModeld& model)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(path.string());

    for(auto& node : root)
    {
        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            if(it->first.as<string>() == "extrinsic")
            {
                Extrinsicsd extrinsics = loadMatrixFromYAML<double, 4, 4>(it);
                model.setExtrinsics(extrinsics);
            }
            else if(it->first.as<string>() == "intrinsic")
            {
                Intrinsicsd intrinsics = loadMatrixFromYAML<double, 3, 3>(it);
                model.setIntrinsics(intrinsics);
            }
            else if(it->first.as<string>() == "distortion")
            {
                Distortiond distortion = loadMatrixFromYAML<double, 6, 1>(it);
                model.setDistortion(distortion);
            }
            else if(it->first.as<string>() == "extrinsic_estimate")
            {
                Extrinsicsd extrinsics = loadMatrixFromYAML<double, 4, 4>(it);
                model.setExtrinsicsEstimate(extrinsics);
            }
        }
    }   
}

bool loadScanImageFromDirectory(
    const boost::filesystem::path& path, 
    ScanImage& image, 
    const size_t& positionNr, const size_t& imageNr)
{
    // Convert position and image number to strings
    stringstream pos_str;
    pos_str << std::setfill('0') << std::setw(5) << positionNr;

    stringstream image_str;
    pos_str << std::setfill('0') << std::setw(5) << imageNr;

    // Construct a path to image directory and check
    boost::filesystem::path scan_image_dir = path / "scan_images" / pos_str.str();
    if(boost::filesystem::exists(scan_image_dir))
    {
        boost::filesystem::path meta_path = scan_image_dir / image_str.str() / ".yaml";
        boost::filesystem::path image_path = scan_image_dir / image_str.str() / ".png";

        std::cout << timestamp << "Loading " << image_path << std::endl;
        image.image = cv::imread(image_path.string(), 1);

        std::cout << timestamp << "Loading " << meta_path << std::endl;
        loadPinholeModelFromYAML(meta_path, image.camera);
    }
    else
    {
        std::cout << timestamp << "Warning: Image directory does not exist: "
                  << scan_image_dir << std::endl;
        return false;
    }

    return true;
}

void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr)
{  
    // Save scan data
    std::cout << timestamp << "Saving scan postion " << positionNr << std::endl;
    if(position.scan)
    {
        saveScanToDirectory(path, *position.scan, positionNr);
    }
    else
    {
        std::cout << timestamp << "Warning: Scan position " << positionNr
                  << " contains no scan data." << std::endl;
    }
    
    // Save scan images
    for(size_t i = 0; i < position.images.size(); i++)
    {
        saveScanImageToDirectory(path, *position.images[i], positionNr, i);
    }
}

void get_all(
    const boost::filesystem::path& root, 
    const string& ext, vector<boost::filesystem::path>& ret)
{
    if(!boost::filesystem::exists(root) || !boost::filesystem::is_directory(root))
    {
        return;
    } 

    boost::filesystem::directory_iterator it(root);
    boost::filesystem::directory_iterator endit;

    while(it != endit)
    {
        if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ext)
        {
            ret.push_back(it->path().filename());
        } 
        ++it;
    }
}

bool loadScanPositionFromDirectory(const boost::filesystem::path& path, ScanPosition& position, const size_t& positionNr)
{
    bool scan_read = false;
    bool images_read = false;

    std::cout << timestamp << "Loading scan position " << positionNr << std::endl;
    if(!loadScanFromDirectory(path, *position.scan, positionNr, true))
    {
        std::cout << timestamp << "Warning: Scan position " << positionNr 
                  << " does not contain scan data." << std::endl;
    }

    boost::filesystem::path img_directory = path / "scan_images";
    if(boost::filesystem::exists(img_directory))
    {
        // Find all .png and .yaml files
        vector<boost::filesystem::path> image_files;
        vector<boost::filesystem::path> meta_files;

        get_all(img_directory, ".png", image_files);
        get_all(img_directory, ".yaml", meta_files);

        if(meta_files.size() == image_files.size())
        {
            std::sort(meta_files.begin(), meta_files.end());
            std::sort(image_files.begin(), image_files.end());

            for(size_t i = 0; i < meta_files.size(); i++)
            {
                ScanImagePtr img(new ScanImage);
                loadScanImageFromDirectory(path, *img, positionNr, i);
                position.images.push_back(img);
            }
        }
        else
        {
            std::cout << timestamp << "Warning: YAML / Image count mismatch." << std::endl;
            return false;
        }

    }
    else
    {
        std::cout << timestamp << "Scan position " << positionNr 
                  << " has no images." << std::endl;
    }
    return true;
}

void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& project)
{
    YAML::Node yaml;
    saveMatrixToYAML<double, 4, 4>(yaml, "position", project.pose);

    std::ofstream out(path.c_str());
    if (out.good())
    {
        out << yaml;
    }
    else
    {
        std::cout << timestamp << "Warning: Unable to open " 
                  << path.string() << "for writing." << std::endl;
    }

    for(size_t i = 0; i < project.positions.size(); i++)
    {
        saveScanPositionToDirectory(path, *project.positions[i], i);
    }
}

bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& project)
{
    // Iterate over all subdirectories

    return true;
}

} // namespace lvr2
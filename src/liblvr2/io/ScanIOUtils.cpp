#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PLYIO.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>



namespace lvr2
{

void writeScanMetaYAML(const boost::filesystem::path& path, const Scan& scan)
{
    YAML::Node meta;

    // Write start and end time
    meta["start_time"]  = scan.m_startTime;
    meta["end_time"]    = scan.m_endTime;

    // Write pose estimation and registration
    const double* poseData = scan.m_poseEstimation.data();
    const double* registrationData = scan.m_registration.data();
    for(size_t i = 0; i < 16; i++)
    {
        meta["pose_estimate"].push_back(poseData[i]);
        meta["registration"].push_back(registrationData[i]);
    }

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
        std::cout << timestamp << "Warning: Unable to open " << path.string() << "for reading." << std::endl;
    }
    
}

void saveScanToDirectory(const boost::filesystem::path& path, const Scan& scan, const size_t& positionNr)
{
    // Create full path from root and scan number
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << positionNr << endl;
    boost::filesystem::path directory = path / ss.str();
    
    // Check if directory exists, if not create it and write meta data
    // and into the new directory
    if(!boost::filesystem::exists(path))
    {
        boost::filesystem::create_directory(directory);        

        // Create yaml file with meta information
        std::cout << "Creating meta.yaml..." << std::endl;
        

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
            std::cout << "Warning: Point cloud pointer is empty!" << std::endl;
        }
    }
    else
    {
        std::cout << timestamp 
                  << "Warning: Directory " << path 
                  << " already exists. Will not override..." << std::endl;
    }
    
}

bool loadScanFromDirectory(const boost::filesystem::path& path, Scan& scan, const size_t& positionNr)
{
    if(boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
    {

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

template<typename T, int Rows, int Cols>
Eigen::Matrix<T, Rows, Cols> loadMatrixFromYAML(const YAML::const_iterator& it)
{
    // Alloc memory for matrix entries
    T data[Rows * Cols] = {0};

    // Read entries
    int c = 0;
    for (auto& i : it->second)
    {
        if(c < Rows * Cols)
        {
            data[c++] = i.as<T>();
        }
        else
        {
            std::cout << timestamp << "Warning: Load Matrix from YAML: Buffer overflow." << std::endl;
            break;
        }
    }
    return Eigen::Map<Eigen::Matrix<T, Rows, Cols>>(data);
}

void saveScanToHDF5(const std::string filename, const size_t& positionNr)
{

}

bool loadScanFromHDF5(const std::string filename, const size_t& positionNr)
{
    return true;
}

void saveScanImageToDirectory(const boost::filesystem::path& path, const ScanImage& image, const size_t& positionNr)
{

}

bool loadScanImageFromDirectory(const boost::filesystem::path& path, ScanImage& image, const size_t& positionNr)
{
    return true;
}

void saveScanPositionToDirectory(const boost::filesystem::path& path, const ScanPosition& position, const size_t& positionNr)
{

}

bool loadScanPositionFromDirectory(const boost::filesystem::path& path, ScanPosition& position, const size_t& positionNr)
{
    return true;
}

void saveScanProjectToDirectory(const boost::filesystem::path& path, const ScanProject& position, const size_t& positionNr)
{

}

bool loadScanProjectFromDirectory(const boost::filesystem::path& path, ScanProject& position, const size_t& positionNr)
{
    return true;
}



} // namespace lvr2
#include <yaml-cpp/yaml.h>
#include "lvr2/io/PlutoMetaDataIO.hpp"

namespace lvr2
{
size_t PlutoMetaDataIO::readSpectralMetaData(const boost::filesystem::path &fn, floatArr &angles)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(fn.string());
    size_t size = 0;
    for (auto &n : root)
    {
        angles = floatArr(new float[n.size()]);
//        std::cout << n.size() << std::endl;
        size = n.size();
        for (YAML::const_iterator it = n.begin(); it != n.end(); ++it)
        {
            // not sorted. key as index.
            angles[it->first.as<int>()] = it->second["angle"].as<float>();
        }
    }

    return size;
}

void PlutoMetaDataIO::readScanMetaData(const boost::filesystem::path &fn, ScanData &scan)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(fn.string());
    for (auto &n : root)
    {
        for (YAML::const_iterator it = n.begin(); it != n.end(); ++it)
        {
            if (it->first.as<string>() == "Start")
            {
                // Parse start time
                float sec = it->second["sec"].as<float>();
                float nsec = it->second["nsec"].as<float>();

                std::cout << "Start: " << sec << "; " << nsec << std::endl;
            }
            else if (it->first.as<string>() == "End")
            {
                // Parse end time
                float sec = it->second["sec"].as<float>();
                float nsec = it->second["nsec"].as<float>();

                std::cout << "End: " << sec << "; " << nsec << std::endl;
            }
            else if (it->first.as<string>() == "Pose")
            {
                // Parse Position
                if (it->second["Position"])
                {
                    YAML::Node tmp = it->second["Position"];
                    float x = tmp["x"].as<float>();
                    float y = tmp["y"].as<float>();
                    float z = tmp["z"].as<float>();

                    std::cout << "Pos: " << x << ", " << y << ", " << z << std::endl;
                }
                if (it->second["Rotation"])
                {
                    YAML::Node tmp = it->second["Rotation"];
                    float x = tmp["x"].as<float>();
                    float y = tmp["y"].as<float>();
                    float z = tmp["z"].as<float>();

                    std::cout << "Rot: " << x << ", " << y << ", " << z << std::endl;
                }
            }
            else if (it->first.as<string>() == "Config")
            {
                // Parse Angles
                if (it->second["Theta"])
                {
                    YAML::Node tmp = it->second["Theta"];
                    float min = tmp["min"].as<float>();
                    float max = tmp["max"].as<float>();

                    scan.m_vFieldOfView = max - min;
                    scan.m_vResolution = tmp["delta"].as<float>();
                    std::cout << "T: " << scan.m_vFieldOfView << "; " << scan.m_vResolution << std::endl;
                }
                if (it->second["Phi"])
                {
                    YAML::Node tmp = it->second["Phi"];
                    float min = tmp["min"].as<float>();
                    float max = tmp["max"].as<float>();

                    scan.m_hFieldOfView = max - min;
                    scan.m_hResolution = tmp["delta"].as<float>();
                    std::cout << "P: " << scan.m_hFieldOfView << "; " << scan.m_hResolution << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
    return;
}

} // namespace lvr2

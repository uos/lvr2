#include "lvr2/io/descriptions/MetaFormatFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/Util.hpp"
#include "lvr2/io/yaml/ScanPosition.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <unordered_set>

namespace lvr2
{

bool isMetaFile(const std::string& filename)
{
    boost::filesystem::path p(filename);
    std::unordered_set<std::string> metaExtensions = {".yaml", ".slam6d", ".frames" };
    return metaExtensions.find(p.extension().string()) != metaExtensions.end();
}

void saveMetaInformation(const std::string &outfile, const YAML::Node &node)
{
    boost::filesystem::path p(outfile);

    if(p.extension() == "")
    {
        p += ".yaml";
    }

    if (p.extension() == ".yaml")
    {
        std::ofstream out(p.string().c_str());
        out << node;
        out.close();
    }
    else if (p.extension() == ".slam6d")
    {
        boost::filesystem::path dir = p.parent_path();
        std::string filename = p.stem().string();
        boost::filesystem::path posePath = dir / (filename + ".pose");
        boost::filesystem::path framesPath = dir / (filename + ".frames");

        // ScanPosition sp;
        // sp = node;

        // Eigen::Affine3d T(sp.transformation);

        // Try to get pose estimation from yaml node
        // and write it to pose file in the directory
        // encoded in the pseudo meta path
        if (node["pose_estimation"])
        {
            // TODO: check this
            // 1. the pose calculation seems to be wrong. use eigen like conversions to euler angles and translation instead
            // 2. need to transform in cm and degrees

            Transformf transform = node["pose_estimation"].as<Transformf>();
            BaseVector<float> position;
            BaseVector<float> angles;
            transform.transposeInPlace();

            getPoseFromMatrix(position, angles, transform);

            std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << posePath << std::endl;
            std::cout << transform << std::endl;
            writePose(position, angles, posePath);
        }

        // Same for registration. If present, write frames file
        if (node["registration"])
        {
            Transformf transform = node["registration"].as<Transformf>();
            writeFrame(transform, framesPath);
        }

        // Or transformation. If present, write frames file
        if (node["transformation"])
        {
            Transformf transform = node["transformation"].as<Transformf>();
            std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << framesPath << std::endl;
            std::cout << transform << std::endl;
            writeFrame(transform, framesPath);
        }
    } else {
        std::cout << timestamp << " [MetaFormatFactory] Meta extension " << p.extension() << " unknown. " << std::endl; 
    }
}

YAML::Node loadMetaInformation(const std::string &in)
{
    boost::filesystem::path inPath(in);

    if(inPath.extension() == "")
    {
        inPath += ".yaml";
    }

    if (inPath.extension() == ".yaml")
    {
        YAML::Node n;
        if (boost::filesystem::exists(inPath))
        {
            // std::cout << timestamp
            //           << "LoadMetaInformation(YAML): Loading " << inPath << std::endl;
            n = YAML::LoadFile(inPath.string());
        }
        else
        {
            std::cout << timestamp
                      << "LoadMetaInformation(YAML): Unable to find yaml file: " << inPath << std::endl;
        }
        return n;
    }
    else if (inPath.extension() == ".slam6d")
    {
        YAML::Node node;        

        boost::filesystem::path dir = inPath.parent_path();
        std::string filename = inPath.stem().string();
        boost::filesystem::path posePath = dir / (filename + ".pose");
        boost::filesystem::path framesPath = dir / (filename + ".frames");

        bool pose_exist = boost::filesystem::exists(posePath);
        bool frames_exist = boost::filesystem::exists(framesPath);

        if(!pose_exist && !frames_exist)
        {
            return node;
        }

        // assuming slam6d works on scanPostition level?
        ScanPosition sp;
        node = sp;

        std::ifstream in_str(posePath.c_str());
        if (in_str.good())
        {
            // std::cout << timestamp
            //           << "LoadMetaInformation(SLAM6D): Loading " << poseInPath << std::endl;

            double x, y, z, r, t, p;
            in_str >> x >> y >> z >> r >> t >> p;
            Vector3d pose(x, y, z);
            Vector3d angles(r, t, p);

            Transformd poseEstimate = poseToMatrix(pose, angles);
            node["pose_estimation"] = poseEstimate;
        }
        else
        {
            std::cout << timestamp
                      << "LoadMetaInformation(SLAM6D): Warning: No pose file found." << std::endl;
        }

        if (frames_exist)
        {
            // std::cout << timestamp
            //           << "LoadMetaInformation(SLAM6D): Loading " << framesInPath << std::endl;
            Transformd registration = getTransformationFromFrames<double>(framesPath);
            node["registration"] = registration;
            node["transformation"] = registration;
        }
        else
        {
            // node frames found. taking poseEstimate as transformation
            if(node["pose_estimation"])
            {
                node["transformation"] = node["pose_estimation"].as<Transformd>();
            }
            
            std::cout << timestamp
                      << "LoadMetaInformation(SLAM6D): Warning: No frames file found." << std::endl;
        }

        return node;
    } else {
        std::cout << "Kernel Panic: Meta extension " << inPath.extension() << " unknown. " << std::endl; 
        YAML::Node node;
        return node;
    }
}

} // namespace lvr2
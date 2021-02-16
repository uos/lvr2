#include "lvr2/io/descriptions/MetaFormatFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/Util.hpp"

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
        // Try to get pose estimation from yaml node
        // and write it to pose file in the directory
        // encoded in the pseudo meta path
        if (node["pose_estimate"])
        {
            Transformf transform = node["pose_estimate"].as<Transformf>();
            BaseVector<float> position;
            BaseVector<float> angles;
            getPoseFromMatrix(position, angles, transform);

            //Construct .pose file path and save
            boost::filesystem::path outfilePath(outfile);
            boost::filesystem::path dir = outfilePath.parent_path();
            boost::filesystem::path posePath(outfilePath.stem().string() + ".pose");
            boost::filesystem::path poseOutPath = p.parent_path() / posePath;

            // std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << poseOutPath << std::endl;
            writePose(position, angles, poseOutPath);
        }

        // Same for registration. If present, write frames file
        if (node["registration"])
        {
            Transformf transform = node["registration"].as<Transformf>();
            //Construct .pose file path and save
            boost::filesystem::path outfilePath(outfile);
            boost::filesystem::path dir = outfilePath.parent_path();
            boost::filesystem::path framesPath(outfilePath.stem().string() + ".frames");
            boost::filesystem::path framesOutPath = p.parent_path() / framesPath;
            // std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << framesOutPath << std::endl;
            writeFrame(transform, framesOutPath);
        }

        // Or transformation. If present, write frames file
        if (node["transformation"])
        {
            Transformf transform = node["transformation"].as<Transformf>();
            //Construct .pose file path and save
            boost::filesystem::path outfilePath(outfile);
            boost::filesystem::path dir = outfilePath.parent_path();
            boost::filesystem::path framesPath(outfilePath.stem().string() + ".frames");
            boost::filesystem::path framesOutPath = p.parent_path() / framesPath;
            // std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << framesOutPath << std::endl;
            writeFrame(transform, framesOutPath);
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
        boost::filesystem::path posePath(inPath.stem().string() + ".pose");
        boost::filesystem::path poseInPath = inPath.parent_path() / posePath;
        std::ifstream in_str(poseInPath.c_str());
        if (in_str.good())
        {
            // std::cout << timestamp
            //           << "LoadMetaInformation(SLAM6D): Loading " << poseInPath << std::endl;

            double x, y, z, r, t, p;
            in_str >> x >> y >> z >> r >> t >> p;
            Vector3d pose(x, y, z);
            Vector3d angles(r, t, p);

            Transformd poseEstimate = poseToMatrix(pose, angles);
            node["pose_estimate"] = poseEstimate;
        }
        else
        {
            std::cout << timestamp
                      << "LoadMetaInformation(SLAM6D): Warning: No pose file found." << std::endl;
        }

        boost::filesystem::path framesPath(inPath.stem().string() + ".frames");
        boost::filesystem::path framesInPath = inPath.parent_path() / framesPath;
        if (boost::filesystem::exists(framesInPath))
        {
            // std::cout << timestamp
            //           << "LoadMetaInformation(SLAM6D): Loading " << framesInPath << std::endl;
            Transformd registration = getTransformationFromFrames<double>(framesInPath);
            node["registration"] = registration;
        }
        else
        {
            std::cout << timestamp
                      << "LoadMetaInformation(SLAM6D): Warning: No pose file found." << std::endl;
        }
        return node;
    } else {
        std::cout << "Kernel Panic: Meta extension " << inPath.extension() << " unknown. " << std::endl; 
    }
}

} // namespace lvr2
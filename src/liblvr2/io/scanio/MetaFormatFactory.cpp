#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/MetaFormatFactory.hpp"
#include "lvr2/io/YAML.hpp"
#include "lvr2/util/IOUtils.hpp"
#include "lvr2/util/YAMLUtil.hpp"

#include <unordered_set>

namespace lvr2
{

bool isMetaFile(const std::string& filename)
{
    boost::filesystem::path p(filename);
    std::unordered_set<std::string> metaExtensions = {".yaml", ".slam6d", ".frames" , ".json" , ".pose" };
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

        ScanPosition sp;
        
        if(YAML::convert<ScanPosition>::decode(node, sp))
        {
            // Is scan position
            writePose(sp.poseEstimation, posePath);
            writeFrame(sp.transformation, framesPath);
        } else {
            // what to do here?
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

    if (inPath.extension() == ".yaml" || inPath.extension() == ".json" || inPath.extension() == ".pose")
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
//    else if (inPath.extension() == ".json")
//    {
//        YAML::Node n;
//        boost::filesystem::path dir = inPath.parent_path();
//        std::string filename = inPath.stem().string();
//
//        boost::filesystem::path posePath = dir / (filename + ".pose");
//
//
//        ScanPosition sp;
//
//
//        return n;
//    }

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

        if (pose_exist)
        {
            sp.poseEstimation = getTransformationFromPose<double>(posePath);
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
            sp.transformation = getTransformationFromFrames<double>(framesPath);
        }
        else
        {
            // node frames found. taking poseEstimate as transformation
            sp.transformation = sp.poseEstimation;
            
            std::cout << timestamp
                      << "LoadMetaInformation(SLAM6D): Warning: No frames file found." << std::endl;
        }

        node = sp;

        return node;
    } else {
        std::cout << "Kernel Panic: Meta extension " << inPath.extension() << " unknown. " << std::endl; 
        YAML::Node node;
        return node;
    }
}

} // namespace lvr2
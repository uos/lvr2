#ifndef METAFILEFACTORY_HPP
#define METAFILEFACTORY_HPP

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <iostream>

namespace lvr2
{

void saveMetaInformation(const std::string& outfile, const YAML::node& node) const
{
    boost::filesystem::path p(outfile);

    if(p.extension() == ".yaml")
    {
        std::cout << timestamp << "SaveMetaInformation(YAML): " << outfile << std::endl;
        std::ofstream out(outfile.c_str());
        out << node;
        out.close();
    }
    else if(p.extension == ".slam6d")
    {
        // Try to get pose estimation from yaml node
        // and write it to pose file in the directory
        // encoded in the pseudo meta path 
        if(node["poseEstimate"])
        {
            Transformd transform = node["poseEstimate"].as<Transformd>();
            BaseVector<double> position;
            BaseVector<double> angles;
            getPoseFromMatrix(position, angles, transform);

            //Construct .pose file path and save
            boost::filesystem::path outfilePath(outfile);
            boost::filesystem::path dir = outfilePath.parent_path();
            boost::filesystem::path posePath(outfilePath.stem().string() + ".pose");
            boost::filesystem::path poseOutPath = p.parent_path() / posePath;

            std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << poseOutPath << std::endl;
            writePose(position, angle, poseOutPath);
        }

        if(node["registration"])
        {
             Transformd transform = node["poseEstimate"].as<Transformd>();
              //Construct .pose file path and save
            boost::filesystem::path outfilePath(outfile);
            boost::filesystem::path dir = outfilePath.parent_path();
            boost::filesystem::path framesPath(outfilePath.stem().string() + ".frames");
            boost::filesystem::path framesOutPath = p.parent_path() / framesPath;
            std::cout << timestamp << "SaveMetaInformation(SLAM6D): " << framesOutPath << std::endl;
            writeFrame(transform, poseOutPath);
        }
    }
}

YAML::node loadMetaInformation(const std::string& in, const YAML::node& node) const
{
    boost::filesystem::path inPath(in);
    if(in.extension() == ".yaml")
    {
        YAML::node n;
        if(boost::filesystem::exists(inPath))
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(YAML): Loading " << inPath << std::endl; 
            n = YAML::LoadFile(framesInPath.string());   
        }
        else
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(YAML): Unable to find yaml file: " << inPath << std::endl; 
        }
        return n;
    }
    else if(in.extension() == ".slam6d")
    {
        YAML::node node;

        boost::filesystem::path dir = inPath.parent_path();
        boost::filesystem::path posePath(outfilePath.stem().string() + ".pose");
        boost::filesystem::path poseInPath = inPath.parent_path() / posePath;
        std::ifstream in_str(poseInPath.c_str());
        if(in_str.good())
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(SLAM6D): Loading " << poseInPath << std::endl;

            double x, y, z, r, t, p
            in_str >> x >> y >> z >> r >> t >> p;
            Vector3d pose(x, y, z);
            Vector3d angles(r, t, p);

            Transformd poseEstimate = poseToMatrix(pose, angles);
            node["poseEstimate"] = poseEstimate;
        }
        else
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(SLAM6D): Warning: No pose file found." << std::endl;
        }

        boost::filesystem::path framesPath(outfilePath.stem().string() + ".frames");
        boost::filesystem::path framesInPath = inPath.parent_path() / framesPath;
        if(boost:filesystem::exists(framesInPath))
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(SLAM6D): Loading " << framesInPath << std::endl;
            Transformd registration = getTransformationFromFrames(framesInPath);
            node["registration"] = registration;
        }
        else
        {
            std::cout << timestamp 
                      << "LoadMetaInformation(SLAM6D): Warning: No pose file found."
        }
        return node;
    }
}

} // namespace lvr2

#endif
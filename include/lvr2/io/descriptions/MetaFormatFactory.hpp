#ifndef METAFILEFACTORY_HPP
#define METAFILEFACTORY_HPP

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>

namespace lvr2
{

void saveMetaInformation(const std::string& outfile, const YAML::node& node)
{
    boost::filesystem::path p(outfile);

    if(p.extension() == ".yaml")
    {
        std::cout << timestamp << "SaveMetaInformation(): " << outfile << std::endl;
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

            std::cout << "SaveMetaInformation(): " << poseOutPath << std::endl;
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
            std::cout << "SaveMetaInformation(): " << framesOutPath << std::endl;
            writeFrame(transform, poseOutPath);
        }
    }
}

} // namespace lvr2

#endif
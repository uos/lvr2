#include "Options.hpp"

#include <boost/filesystem.hpp>

#include "lvr2/io/HDF5IO.hpp"
#include "lvr2/display/PointOctree.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/PointBuffer.hpp"

const hdf5tool2::Options* options;
using namespace lvr2;



int main(int argc, char** argv)
{
    boost::filesystem::path inputDir(options->getInputDir());

    if(!boost::filesystem::exists(inputDir))
    {
        std::cout << timestamp << "Error: Directory " << options->getInputDir() << " does not exist" << std::endl;
        exit(-1);
    }
    
    boost::filesystem::path outputPath(options->getOutputDir());

    // Check if output dir exists
    if(!boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "Creating directory " << options->getOutputDir() << std::endl;
        if(!boost::filesystem::create_directory(outputPath))
        {
            std::cout << timestamp << "Error: Unable to create " << options->getOutputDir() << std::endl;
            exit(-1);
        }
    }
    
    outputPath /= options->getOutputFile();
    if(!boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "Error: File exists " << outputPath << std::endl;
        exit(-1);
    }
    
    HDF5IO hdf(outputPath.string(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    

}

#include "Options.hpp"

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/descriptions/ScanProjectSerialization.hpp"
#include "lvr2/io/descriptions/ScanProjectStructureSLAM.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    scanproject_parser::Options options(argc, argv);

    ScanProjectStructureSLAM slam_structure(options.getInputDir());
    DirectoryKernel kernel("");

    ScanProjectPtr project = loadScanProject(slam_structure, kernel);

    std::cout << timestamp << "Program finished" << std::endl;

    return 0;
}

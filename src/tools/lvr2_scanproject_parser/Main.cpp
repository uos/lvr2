#include "Options.hpp"

#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    scanproject_parser::Options options(argc, argv);

    std::string slamDir = options.getInputDir();
    std::string hyperlibDir = "./hyperlib";

    // Read slam6d data from input dir
    DirectorySchemaPtr slamSchemaPtr(new ScanProjectSchemaSLAM);
    DirectoryKernelPtr slamDirKernel(new DirectoryKernel(slamDir));
    DirectoryIO slamIO(slamDirKernel, slamSchemaPtr);
    ScanProjectPtr slamProject = slamIO.loadScanProject();

    // Copy project using slam6d schema
    DirectoryKernelPtr slamCopyKernel(new DirectoryKernel("./slam_copy"));
    DirectoryIO slamIOCopy(slamCopyKernel, slamSchemaPtr);
    slamIOCopy.saveScanProject(slamProject);

    // Copy project using hyperlib schema
    DirectorySchemaPtr hyperlibSchema(new ScanProjectSchemaHyperlib);
    DirectoryKernelPtr slamCopyKernel2(new DirectoryKernel("./hyperlib_copy"));
    DirectoryIO slamIOCopy2(slamCopyKernel2, hyperlibSchema);
    slamIOCopy2.saveScanProject(slamProject);

    return 0;
}

#include "Options.hpp"

#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    scanproject_parser::Options options(argc, argv);

    std::string slamDir = options.getInputDir();
    std::string hyperlibDir = "./hyperlib";

    // Read slam6d data from input dir
    DirectorySchemaPtr slamSchemaPtr(new ScanProjectSchemaSLAM(slamDir));
    DirectoryKernelPtr slamDirKernel(new DirectoryKernel(slamDir));
    DirectoryIO slamIO(slamDirKernel, slamSchemaPtr);
    ScanProjectPtr slamProject = slamIO.loadScanProject();

    std::cout << slamProject->positions[0]->scans.size() << std::endl;

    // Copy project using slam6d schema
    DirectoryKernelPtr slamCopyKernel(new DirectoryKernel("./slam_copy"));
    DirectoryIO slamIOCopy(slamCopyKernel, slamSchemaPtr);
    slamIOCopy.saveScanProject(slamProject);

    // Copy project using hyperlib schema
    std::string hyperlibCopyDir = "./hyperlib_copy";
    DirectorySchemaPtr hyperlibSchema(new ScanProjectSchemaHyperlib(hyperlibCopyDir));
    DirectoryKernelPtr slamCopyKernel2(new DirectoryKernel(hyperlibCopyDir));
    DirectoryIO slamIOCopy2(slamCopyKernel2, hyperlibSchema);
    slamIOCopy2.saveScanProject(slamProject);

    // Copy project into HDF5
    HDF5SchemaPtr hdf5Schema(new ScanProjectSchemaHDF5V2);
    HDF5KernelPtr hdf5Kernel(new HDF5Kernel("slam.h5"));
    HDF5IO hdf5io(hdf5Kernel, hdf5Schema);
    hdf5io.saveScanProject(slamProject);

    return 0;
}

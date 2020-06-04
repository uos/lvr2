#include "Options.hpp"

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/descriptions/ScanProjectSerialization.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    scanproject_parser::Options options(argc, argv);

    ScanProjectSchemaSLAM slam_structure(options.getInputDir());
    ScanProjectSchemaHyperlib hyperlibStructure(options.getOutputDir());
    DirectoryKernel kernel("");

    ScanProjectPtr project = loadScanProject(slam_structure, kernel);

    // Save project in hyperlib structure
    saveScanProject(hyperlibStructure, kernel, project);

    // Save project in slam structure
    ScanProjectSchemaSLAM slam_structure_out("./slam");
    saveScanProject(slam_structure_out, kernel, project);

    // Copy generated scan project directory
    ScanProjectPtr copy_project = loadScanProject(hyperlibStructure, kernel);
    ScanProjectSchemaHyperlib hyperlibStructure_copy(options.getOutputDir() + "_copy");
    saveScanProject(hyperlibStructure_copy, kernel, copy_project);

    std::cout << "HDF5" << std::endl;

    // HDF5Kernel hdf5Kernel_slam("slam.h5");
    // saveScanProject(slam_structure, hdf5Kernel_slam, project);

    HDF5Kernel hdf5Kernel_hyper("hyper.h5");
    saveScanProject(hyperlibStructure, hdf5Kernel_hyper, project);

    std::cout << timestamp << "Program finished" << std::endl;

    return 0;
}

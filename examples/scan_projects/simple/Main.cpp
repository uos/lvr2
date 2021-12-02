#include <iostream>
#include <memory>
#include <lvr2/types/Variant.hpp>

// LOG << "hello world" << std::endl;
#include "../helper/include/Logging.hpp"
// dummyScanProject()
#include "../helper/include/ScanTypesDummies.hpp"
// comparison of scan project entities
#include "../helper/include/ScanTypesCompare.hpp"

// internally used structure for handling sensor data
#include "lvr2/types/ScanTypes.hpp"

// DirectoryIO includes
#include "lvr2/io/scanio/DirectoryIO.hpp"
#include "lvr2/io/scanio/DirectoryKernel.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaRaw.hpp"

// // Hdf5IO includes
// #include "lvr2/io/scanio/HDF5IO.hpp"
// #include "lvr2/io/scanio/HDF5Kernel.hpp"
// #include "lvr2/io/scanio/ScanProjectSchemaHDF5.hpp"

using namespace lvr2;
// // using this is only temporary until old IO is deleted
// using lvr2::scanio::HDF5IO;

void directoryIOExample(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_simple/dirio_data";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(filename));
    DirectoryIO dirio(kernel, schema);

    LOG(Logger::DEBUG) << "Save complete scan project to '" << filename << "'" << std::endl; 
    dirio.save(sp);

    LOG(Logger::DEBUG) << "Load scan project into new buffer" << std::endl;

    // Load scan project again but into a new buffer
    // - alternative: dirio.loadScanProject();
    auto sp_loaded = dirio.ScanProjectIO::load();
    
    // check if the dummy scan scan project equals the saved and loaded scan project
    if(equal(sp, sp_loaded))
    {
        LOG(Logger::DEBUG) << "DirectoryIO saves and loads correctly." << std::endl;
    } else {
        LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
    }

    LOG(Logger::DEBUG) << "Take a look at '" << filename << "'" << std::endl;
    LOG(Logger::DEBUG) << "- You can use 'tree' to show the entire directory structure" << std::endl;
}

// void hdf5IOExample(ScanProjectPtr sp)
// {
//     // specify a hdf5 file to store the data into
//     std::string filename = "examples_sp_simple/hdf5io_data.h5";

//     // create IO
//     // - Kernel: Hdf5
//     // - Schema: Hdf5
//     HDF5KernelPtr kernel(new HDF5Kernel(filename));
//     HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());
//     HDF5IO hdf5io(kernel, schema);

//     LOG(Logger::DEBUG) << "Save complete scan project to '" << filename << "'" << std::endl; 
//     hdf5io.save(sp);

//     LOG(Logger::DEBUG) << "Load scan project into new buffer" << std::endl;
//     // Load scan project again but into a new buffer
//     // - alternative: hdf5io.loadScanProject();
//     auto sp_loaded = hdf5io.ScanProjectIO::load();
    
//     // check if the dummy scan scan project equals the saved and loaded scan project
//     if(equal(sp, sp_loaded))
//     {
//         LOG(Logger::DEBUG) << "Hdf5IO saves and loads correctly." << std::endl;
//     } else {
//         LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
//     }

//     LOG(Logger::DEBUG) << "Take a look at '" << filename << "'" << std::endl;
//     LOG(Logger::DEBUG) << "- You can use 'HDFCompass' to view the entire hdf5 structure" << std::endl;
// }


int main(int argc, char** argv)
{
    LOG.setLoggerLevel(Logger::DEBUG);

    LOG(Logger::HIGHLIGHT) << "ScanProjects Simple" << std::endl;
    // generate 

    LOG(Logger::DEBUG) << "Generating dataset, wait." << std::endl;
    ScanProjectPtr sp = dummyScanProject();

    LOG(Logger::INFO) << "1. Example: DirectoryIO" << std::endl;
    LOG.tab();
    directoryIOExample(sp);
    LOG.deltab();

    // LOG(Logger::INFO) << "2. Example: Hdf5IO" << std::endl; 
    // LOG.tab();
    // hdf5IOExample(sp);
    // LOG.deltab();

    //// Further comments:
    // Hdf5IO takes longer to store and read because of internal compressions
    // this makes the hdf5 file smaller compared to the directory structure
    // if you want more speed you can adjust the compression level as 
    // shown in the compression example

    return 0;
}
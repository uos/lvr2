#include <iostream>
#include <memory>

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

// Hdf5IO includes
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/io/scanio/HDF5Kernel.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaHDF5.hpp"

using namespace lvr2;
// using this is only temporary until old IO is deleted
using lvr2::scanio::HDF5IO;


void hdf5IOCompressedExample(ScanProjectPtr sp, size_t compressionLevel)
{
    // specify a hdf5 file to store the data into
    std::string filename = "examples_sp_compression/sp_compressed_" + std::to_string(compressionLevel) + ".h5";

    // set compression level to max
    HDF5KernelConfig config;
    config.compressionLevel = compressionLevel;

    // create IO
    // - Kernel: Hdf5
    // - Schema: Hdf5
    HDF5KernelPtr kernel(new HDF5Kernel(filename, config));
    HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());
    HDF5IO hdf5io(kernel, schema);

    LOG(Logger::DEBUG) << "Save complete scan project compressed to '" << filename << "'" << std::endl; 
    hdf5io.save(sp);

    LOG(Logger::DEBUG) << "Load scan project into new buffer" << std::endl;
    // Load scan project again but into a new buffer
    // - alternative: hdf5io.loadScanProject();
    auto sp_loaded = hdf5io.ScanProjectIO::load();
    
    // check if the dummy scan scan project equals the saved and loaded scan project
    if(equal(sp, sp_loaded))
    {
        LOG(Logger::DEBUG) << "Hdf5IO saves and loads correctly." << std::endl;
    } else {
        LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
    }
}

int main(int argc, char** argv)
{
    LOG.setLoggerLevel(Logger::DEBUG);

    LOG(Logger::HIGHLIGHT) << "ScanProjects Simple" << std::endl;
    // generate 
    
    LOG(Logger::DEBUG) << "Generating dataset, wait." << std::endl;
    ScanProjectPtr sp = dummyScanProject();

    LOG(Logger::INFO) << "Writing dummy scan project with different levels of compression" << std::endl;

    for(size_t i=0; i<10; i++)
    {
        LOG(Logger::HIGHLIGHT) << "Compression Level: " << i << std::endl;
        LOG.tab();
        hdf5IOCompressedExample(sp, i);
        LOG.deltab();
    }

    //// Further comments:
    // My tests with compression levels
    // 'du -sh *' output:

    // chunk: {numElements, 1}
    // 586M	examples_sp_compression/sp_compressed_0.h5
    // 55M	examples_sp_compression/sp_compressed_1.h5
    // 54M	examples_sp_compression/sp_compressed_2.h5
    // 54M	examples_sp_compression/sp_compressed_3.h5
    // 52M	examples_sp_compression/sp_compressed_4.h5
    // 52M	examples_sp_compression/sp_compressed_5.h5
    // 52M	examples_sp_compression/sp_compressed_6.h5
    // 52M	examples_sp_compression/sp_compressed_7.h5
    // 51M	examples_sp_compression/sp_compressed_8.h5
    // 51M	examples_sp_compression/sp_compressed_9.h5


    return 0;
}
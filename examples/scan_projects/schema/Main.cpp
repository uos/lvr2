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

////////////////////////////////
/// SCHEMA USAGE 1
////////////////////////////////
void useScanProjectSchemaRaw(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_schema/schema_raw";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(filename));
    DirectoryIO dirio(kernel, schema);

    dirio.save(sp);

    // Load scan project again but into a new buffer
    // - alternative: dirio.loadScanProject();
    auto sp_loaded = dirio.ScanProjectIO::load();
    
    // check if the dummy scan scan project equals the saved and loaded scan project
    if(!equal(sp, sp_loaded))
    {
        LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
    }
}

////////////////////////////////
/// SCHEMA USAGE 2
////////////////////////////////
class MySchema : public ScanProjectSchemaRaw 
{
public:
    // Using constructor of base class
    using ScanProjectSchemaRaw::ScanProjectSchemaRaw;

    // change the scan function such that its dataset is stored into "points.ply" completely
    virtual Description scan(const size_t& scanPosNo, const size_t& lidarNo, const size_t& scanNo) const {
        Description d_scan = ScanProjectSchemaRaw::scan(scanPosNo, lidarNo, scanNo);
        // store entire scan into "points.ply"
        d_scan.data = "points.ply";
        return d_scan;
    }
};

void useMySchema(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_schema/my_schema";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new MySchema(filename));
    DirectoryIO dirio(kernel, schema);

    dirio.save(sp);

    // Load scan project again but into a new buffer
    // - alternative: dirio.loadScanProject();
    auto sp_loaded = dirio.ScanProjectIO::load();
    
    // check if the dummy scan scan project equals the saved and loaded scan project
    if(!equal(sp, sp_loaded))
    {
        LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
    }
}

////////////////////////////////
/// SCHEMA USAGE 3
////////////////////////////////
class MySchemaAdvanced : public ScanProjectSchemaRaw 
{
public:
    // Using constructor of base class
    using ScanProjectSchemaRaw::ScanProjectSchemaRaw;

    // We want to save specific scan channels in "points.ply" such as "points", "normals", "colors".
    // Thus, we can visualize the pointclouds by well known viewer-software
    // The difference to "MySchema" is, that we dont discard the other channels that cannot be stored in a "ply" file
    virtual Description scanChannel(const size_t& scanPosNo, const size_t& lidarNo, const size_t& scanNo, const std::string& channelName) const {
        Description d_scan_channel = ScanProjectSchemaRaw::scanChannel(scanPosNo, lidarNo, scanNo, channelName);
        
        // example: store only the points and colors in "points.ply"
        // - normals are kept in a binary buffer beside the ply

        if(channelName == "points" || channelName == "colors")
        {
            *d_scan_channel.dataRoot += "/points.ply";
        }

        return d_scan_channel;
    }
};

void useMySchemaAdvanced(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_schema/my_schema_advanced";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new MySchemaAdvanced(filename));
    DirectoryIO dirio(kernel, schema);

    dirio.save(sp);

    // Load scan project again but into a new buffer
    // - alternative: dirio.loadScanProject();
    auto sp_loaded = dirio.ScanProjectIO::load();
    
    // check if the dummy scan scan project equals the saved and loaded scan project
    if(!equal(sp, sp_loaded))
    {
        LOG(Logger::WARNING) << "Something went wrong. Saved and loaded scan project are not equal" << std::endl;
    }
}

int main(int argc, char** argv)
{
    LOG.setLoggerLevel(Logger::DEBUG);

    LOG(Logger::HIGHLIGHT) << "ScanProjects Schema" << std::endl;
    // generate 

    LOG(Logger::DEBUG) << "Generating dataset, wait." << std::endl;
    ScanProjectPtr sp = dummyScanProject();

    LOG(Logger::HIGHLIGHT) << "1. Example: ScanProjectSchemaRaw" << std::endl;
    LOG.tab();
    useScanProjectSchemaRaw(sp);
    LOG.deltab();

    LOG(Logger::HIGHLIGHT) << "2. Example: MySchema" << std::endl;
    LOG.tab();
    useMySchema(sp);
    LOG.deltab();

    LOG(Logger::HIGHLIGHT) << "3. Example: MySchemaAdvanced" << std::endl;
    LOG.tab();
    useMySchemaAdvanced(sp);
    LOG.deltab();

    return 0;
}
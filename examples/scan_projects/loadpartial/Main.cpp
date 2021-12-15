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

void loadPartial(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_loadpartial/dirio_data";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(filename));
    DirectoryIO dirio(kernel, schema);

    LOG(Logger::DEBUG) << "Save complete scan project to '" << filename << "'" << std::endl; 
    dirio.save(sp);

    std::cout << std::endl;
    LOG(Logger::INFO) << "Load all datasets of Hierarchy level ScanProject" << std::endl;
    // Hierarchy level 0
    auto sp_loaded = dirio.ScanProjectIO::load();

    // count total points
    size_t total_points = 0;
    for(auto pos : sp_loaded->positions)
    {
        for(auto lidar : pos->lidars)
        {
            for(auto scan : lidar->scans)
            {
                total_points += scan->points->numPoints();
            }
        }
    }

    LOG(Logger::DEBUG) << "Total points: " << total_points << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load all datasets of Hierarchy level ScanPosition" << std::endl;
    // Example load scan position of id = 0
    ScanPositionPtr pos = dirio.ScanPositionIO::load(0);

    // count total points
    total_points = 0;
    for(auto lidar : pos->lidars)
    {
        for(auto scan : lidar->scans)
        {
            total_points += scan->points->numPoints();
        }
    }
    LOG(Logger::DEBUG) << "Total points: " << total_points << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load all datasets of Hierarchy level Sensor" << std::endl;
    // Load lidar 0 at scan position 0
    LIDARPtr lidar = dirio.LIDARIO::load(0,0);
    
    // count total points
    total_points = 0;
    for(auto scan : lidar->scans)
    {
        total_points += scan->points->numPoints();
    }
    LOG(Logger::DEBUG) << "Total points: " << total_points << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load all datasets of Hierarchy level SensorData" << std::endl;
    // Load scan 0 at lidar 0 at scan position 0
    ScanPtr scan = dirio.ScanIO::load(0, 0, 0);
    LOG(Logger::DEBUG) << "Total points: " << scan->points->numPoints() << std::endl;
    std::cout << std::endl;

    LOG.deltab();
    LOG.deltab();
    LOG.deltab();
}

void loadPartialMeta(ScanProjectPtr sp)
{
    // specify a directory to store the data into
    std::string filename = "examples_sp_loadpartial/dirio_data";

    // create IO
    // - Kernel: Directory
    // - Schema: Raw
    DirectoryKernelPtr kernel(new DirectoryKernel(filename));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(filename));
    DirectoryIO dirio(kernel, schema);

    LOG(Logger::DEBUG) << "Save complete scan project to '" << filename << "'" << std::endl; 
    dirio.save(sp);


    LOG(Logger::INFO) << "Load meta of Hierarchy level ScanProject" << std::endl;
    // Hierarchy level 0
    boost::optional<YAML::Node> meta = dirio.ScanProjectIO::loadMeta();
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load meta of Hierarchy level ScanPosition" << std::endl;
    // Example load scan position of id = 0
    meta = dirio.ScanPositionIO::loadMeta(0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load meta of Hierarchy level Sensor (LIDAR)" << std::endl;
    // Load lidar 0 at scan position 0
    meta = dirio.LIDARIO::loadMeta(0,0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load meta of Hierarchy level SensorData (Scan)" << std::endl;
    // Load scan 0 at lidar 0 at scan position 0
    meta = dirio.ScanIO::loadMeta(0, 0, 0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;


    std::unordered_map<std::string, YAML::Node> channel_metas = dirio.ScanIO::loadChannelMetas(0, 0, 0);
    for(auto elem : channel_metas)
    {
        LOG(Logger::DEBUG) << elem.first << ":" << std::endl;
        LOG.tab();
        LOG(Logger::DEBUG) << elem.second << std::endl;
        LOG.deltab();
    }
    std::cout << std::endl;

    LOG.deltab();

    //// CAMERA

    LOG(Logger::INFO) << "Load meta of Hierarchy level Sensor (Camera)" << std::endl;
    // Load lidar 0 at scan position 0
    meta = dirio.CameraIO::loadMeta(0,0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load meta of Hierarchy level SensorData (CameraImage)" << std::endl;
    // Load scan 0 at lidar 0 at scan position 0
    meta = dirio.CameraImageIO::loadMeta(0, 0, 0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.deltab();


    /// HYPERSPECTRAL CAMERA

    LOG(Logger::INFO) << "Load meta of Hierarchy level Sensor (HyperspectralCamera)" << std::endl;
    // Load lidar 0 at scan position 0
    meta = dirio.HyperspectralCameraIO::loadMeta(0,0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();
    LOG(Logger::INFO) << "Load meta of Hierarchy level SensorData (HyperspectralCameraPanorama)" << std::endl;
    // Load scan 0 at lidar 0 at scan position 0
    meta = dirio.HyperspectralPanoramaIO::loadMeta(0, 0, 0);
    LOG(Logger::DEBUG) << meta << std::endl;
    std::cout << std::endl;

    LOG.tab();

    auto channel_meta = dirio.HyperspectralPanoramaChannelIO::loadMeta(0,0,0,0);
    LOG(Logger::DEBUG) << channel_meta << std::endl;
    std::cout << std::endl;

    LOG.deltab();


    LOG.deltab();
    LOG.deltab();
    LOG.deltab();
}

int main(int argc, char** argv)
{

    if(argc > 1)
    {
        std::string filename = argv[1];
        HDF5KernelPtr kernel(new HDF5Kernel(filename));
        HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());
        HDF5IO hdf5io(kernel, schema);

        std::cout << timestamp << "Load ScanProject" << std::endl;
        auto sp_loaded = hdf5io.ScanProjectIO::load();
        std::cout << timestamp << "Done." << std::endl;



        ReductionAlgorithmPtr red(new FixedSizeReductionAlgorithm(1000));

        if(sp_loaded)
        {
            // Get Scan 0 of Lidar 0 of Scan Position 0
            ScanPtr scan = sp_loaded->positions[0]->lidars[0]->scans[0];
        
            if(scan)
            {
                std::cout << timestamp << "Load " << scan->numPoints << " points completely" << std::endl;
                scan->load();
                std::cout << timestamp << "Done." << std::endl;
                std::cout << *scan->points << std::endl;

                scan->unload();
                std::cout << timestamp << "Load " << scan->numPoints << " points reduced" << std::endl;
                scan->load(red);
                std::cout << timestamp << "Done." << std::endl;
                std::cout << *scan->points << std::endl;
            }

            // Get Camera 0 of Scan position 0
            CameraPtr cam = sp_loaded->positions[0]->cameras[0];

            // Get Image 0 of Group 0
            if(cam->images[0].is_type<CameraImageGroupPtr>() )
            {
                CameraImageGroupPtr group;
                group <<= cam->images[0];
                CameraImagePtr img;
                img <<= group->images[0];

                if(img)
                {
                    std::cout << timestamp << "Load image completely" << std::endl;
                    img->image = img->image_loader();
                    std::cout << timestamp << "Done." << std::endl;

                    cv::namedWindow("image", cv::WINDOW_NORMAL);
                    cv::imshow("image", img->image);
                    cv::waitKey(0);
                }
            }
        }
        

    } else {

        LOG.setLoggerLevel(Logger::DEBUG);

        LOG(Logger::HIGHLIGHT) << "ScanProjects Load Partial" << std::endl;
        // generate 

        LOG(Logger::DEBUG) << "Generating dataset, wait." << std::endl;
        ScanProjectPtr sp = dummyScanProject();

        std::cout << std::endl;
        LOG(Logger::HIGHLIGHT) << "1. Example: Load datasets partially" << std::endl;
        LOG.tab();
        loadPartial(sp);
        LOG.deltab();

        std::cout << std::endl;
        LOG(Logger::HIGHLIGHT) << "2. Example: Load meta information partially" << std::endl; 
        LOG.tab();
        loadPartialMeta(sp);
        LOG.deltab();
    }


    //// Further comments:
    

    return 0;
}
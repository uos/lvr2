#include "Logging.hpp"

#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"


#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"

#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

// #include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
// #include "lvr2/io/hdf5/ScanProjectIO.hpp"


#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include <boost/filesystem.hpp>

#include "lvr2/util/Synthetic.hpp"

#include <boost/type_index.hpp>

#include <unordered_map>
#include <unordered_set>

#include "Hdf5ReaderOld.hpp"
#include "ScanTypesCompare.hpp"

using namespace lvr2;

ScanProjectPtr dummyScanProject()
{
    ScanProjectPtr ret(new ScanProject);

    for(size_t i=0; i<5; i++)
    {
        ScanPositionPtr scan_pos(new ScanPosition);
        scan_pos->transformation = Transformd::Identity();
        scan_pos->transformation(0,3) = static_cast<double>(i);

        for(size_t j=0; j<2; j++)
        {
            LIDARPtr lidar(new LIDAR);
            lidar->name = "Riegl";
            lidar->transformation = Transformd::Identity();
            lidar->transformation(1,3) = static_cast<double>(j);

            for(size_t k=0; k<10; k++)
            {
                ScanPtr scan(new Scan);

                scan->hResolution = 1.0;
                scan->vResolution = 0.2;
                scan->thetaMin = -M_PI;
                scan->thetaMax = M_PI;

                scan->phiMin = -M_PI;
                scan->phiMax = M_PI;
                scan->poseEstimation = lvr2::Transformd::Identity();
                scan->transformation = lvr2::Transformd::Identity();
                scan->transformation(2,3) = static_cast<double>(k);

                scan->points = synthetic::genSpherePoints(50, 50);

                scan->numPoints = scan->points->numPoints();
                scan->startTime = 0.0;
                scan->endTime  = 100.0;

                lidar->scans.push_back(scan);
            }
            
            scan_pos->lidars.push_back(lidar);
        }

        // for(size_t j=0; j<2; j++)
        // {
        //     CameraPtr scan_cam(new Camera);
        //     scan_cam->transformation = Transformd::Identity();
        //     scan_cam->transformation(1,3) = -static_cast<double>(j);
        //     scan_cam->model.distortionModel = "opencv";
        //     scan_cam->model.k.resize(10);
        //     scan_cam->model.cx = 100.2;
        //     scan_cam->model.cy = 50.5;
        //     scan_cam->model.fx = 120.99;
        //     scan_cam->model.fy = 90.72;

        //     for(size_t k=0; k<10; k++)
        //     {
        //         scan_cam->model.k[k] = static_cast<double>(k) / 4.0;
        //     }
            
        //     for(size_t k=0; k<7; k++)
        //     {
        //         CameraImagePtr si = synthetic::genLVRImage();
        //         si->timestamp = 0.0;
        //         si->transformation = Transformd::Identity();
        //         si->transformation(2,3) = -static_cast<double>(k);
        //         si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(k + 1);
        //         scan_cam->images.push_back(si);
        //     }

        //     scan_cam->name = "Canon";
        //     scan_pos->cameras.push_back(scan_cam);
        // }

        // for(size_t j=0; j<2; j++)
        // {
        //     HyperspectralCameraPtr h_cam(new HyperspectralCamera);

        //     h_cam->transformation = Transformd::Identity();
        //     h_cam->transformation(1,3) = -static_cast<double>(j);

        //     h_cam->model.principal(0) =  5.5;
        //     h_cam->model.principal(1) = 4.4;

        //     h_cam->model.focalLength(0) = 10.1;
        //     h_cam->model.focalLength(1) = 10.2;

        //     h_cam->model.distortion.resize(3);
        //     h_cam->model.distortion[0] = 2.0;
        //     h_cam->model.distortion[1] = 1.0;
        //     h_cam->model.distortion[2] = 0.5;

        //     for(size_t k=0; k<3; k++)
        //     {
        //         HyperspectralPanoramaPtr pano(new HyperspectralPanorama);

        //         pano->resolution[0] = 200;
        //         pano->resolution[1] = 200;
        //         pano->wavelength[0] = 100.0;
        //         pano->wavelength[1] = 900.25;

        //         for(size_t l=0; l<7; l++)
        //         {
        //             HyperspectralPanoramaChannelPtr hchannel(new HyperspectralPanoramaChannel);

        //             CameraImagePtr si = synthetic::genLVRImage();
        //             hchannel->channel = si->image.clone();
        //             hchannel->timestamp = 0.0;
        //             pano->channels.push_back(hchannel);
        //         }

        //         h_cam->panoramas.push_back(pano);
        //     }

        //     scan_pos->hyperspectral_cameras.push_back(h_cam);
        // }

        ret->positions.push_back(scan_pos);
    }

    ret->unit = "m";
    ret->coordinateSystem = "right-handed";
    ret->transformation = Transformd::Identity();

    return ret;
}


bool directoryIOTest()
{
    std::string dirname = "scan_project_directory";

    DirectoryKernelPtr kernel2(new DirectoryKernel(dirname));
    DirectorySchemaPtr schema2(new ScanProjectSchemaRaw(dirname));
    DirectoryIO dirio(kernel2, schema2);

    LOG(lvr2::Logger::DEBUG) << "Create Dummy Scanproject..." << std::endl;
    auto sp = dummyScanProject();
    
    LOG(lvr2::Logger::DEBUG) << "Save Scanproject to directory..." << std::endl;
    dirio.save(sp);

    LOG(lvr2::Logger::DEBUG) << "Load Scanproject from directory..." << std::endl;
    auto sp_loaded = dirio.ScanProjectIO::load();


    // Or load partially
    // auto hpchannel = dirio.HyperspectralPanoramaChannelIO::load(0, 0, 0, 0);

    // std::cout << "Loaded channel of size: " << hpchannel->channel.size() << std::endl;


    return equal(sp, sp_loaded);
}

// bool hdf5IOTest()
// {
//     std::string filename = "scan_project.h5";
//     HDF5KernelPtr kernel(new HDF5Kernel(filename));
//     HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());

//     descriptions::HDF5IO hdf5io(kernel, schema);

//     auto sp = dummyScanProject();
//     hdf5io.save(sp);
//     auto sp_loaded = hdf5io.ScanProjectIO::load();

//     return equal(sp, sp_loaded);
// }

// void unitTest()
// {
//     LOG(Logger::INFO) << "Hdf5-IO Test" << std::endl;
    
//     LOG.tab();
//     bool hdf5_success = true;
//     for(size_t i=0; i<10; i++)
//     {
//         if(!hdf5IOTest()) {
//             hdf5_success = false;
//             break;
//         }
//         LOG(Logger::DEBUG) << i << " success." << std::endl;
//     }
//     LOG.deltab();
//     if(hdf5_success)
//     {
//         LOG(Logger::HIGHLIGHT) << "Hdf5-IO success." << std::endl;
//     } else {
//         LOG(Logger::ERROR) << "TODO: Fix Hdf5-IO" << std::endl;
//     }
    
//     std::cout << std::endl;
//     LOG(Logger::INFO) << "Directory-IO Test" << std::endl;
//     LOG.tab();
//     bool dir_success = true;
//     for(size_t i=0; i<10; i++)
//     {
//         if(!directoryIOTest()) {
//             dir_success = false;
//             break;
//         }
//         LOG(Logger::DEBUG) << i << " success." << std::endl;
//     }
//     LOG.deltab();

//     if(dir_success) {
//         LOG(Logger::HIGHLIGHT) << "Directory-IO success." << std::endl;
//     } else {
//         LOG(Logger::ERROR) << "TODO: Fix Directory-IO" << std::endl;
//     }
// }

void loggerTest()
{
    using namespace lvr2;

    // Logger<LoggingLevel::DEBUG> log;

    LOG << "Hallo" << std::endl;

    LOG.setLoggerLevel(lvr2::Logger::DEBUG);

    LOG(lvr2::Logger::ERROR) << "TEEEST" << std::endl;
    LOG.tab();

    for(size_t i=0; i<2; i++)
    {
        LOG(lvr2::Logger::WARNING) << "i: " << i << std::endl;
        // lvr2::cout(lvr2::Logger::WARNING) << "i: " << i << std::endl;
        LOG.tab();
        for(size_t j=0; j<2; j++)
        {
            LOG(lvr2::Logger::INFO) << "j: " << j << std::endl;

            LOG.tab();
            for(size_t k=0; k<2; k++)
            {
                LOG(lvr2::Logger::DEBUG) << "k: " << k << std::endl;
            
                Transformd T = Transformd::Identity();
                LOG(lvr2::Logger::DEBUG) << T << std::endl << std::endl;

                MultiChannelMap m;
                m["hallo"] = Channel<float>(3,3);
                LOG(lvr2::Logger::DEBUG) << m << std::endl;
            }
            LOG.deltab();
        }
        LOG.deltab();
    }
    LOG.deltab();

    LOG(Logger::Level::HIGHLIGHT) << "Finished" << std::endl;
}


void debugTest()
{
    std::string dirname = "bla_bla";

    DirectoryKernelPtr kernel2(new DirectoryKernel(dirname));
    DirectorySchemaPtr schema2(new ScanProjectSchemaRaw(dirname));
    DirectoryIO dirio(kernel2, schema2);

    LOG(lvr2::Logger::DEBUG) << "Create Dummy Scanproject..." << std::endl;
    
    // auto sp = dummyScanProject();


    PointBufferPtr points = lvr2::synthetic::genSpherePoints();

    size_t npoints = points->numPoints();

    // points->erase("points");

    Channel<float> normals(npoints, 3);

    for(size_t i=0; i<npoints; i++)
    {
        normals[i][0] = 1.0;
        normals[i][1] = 0.0;
        normals[i][2] = 0.0;
    }

    Channel<double> some_other_data(npoints, 3);
    
    points->add("normals", normals);
    points->add("some_other_data", some_other_data);
    
    size_t posNo = 0;
    size_t lidarNo = 0;
    size_t scanNo = 0;
    dirio.PointCloudIO::save(posNo, lidarNo, scanNo, points);

    // dirio.PointCloudIO::load(posNo, lidarNo, scanNo);
}

// void compressionTest()
// {
//     // cv::Mat big_image(5000, 3000, CV_8UC3, cv::Scalar(0));

//     cv::Mat big_image = lvr2::synthetic::genLVRImage()->image;

//     cv::resize(big_image, big_image, cv::Size(5000, 3000));

//     int src_type = big_image.type();

//     cv::Mat small_image;
//     cv::resize(big_image, small_image, cv::Size(1000, 500));
//     cv::imshow("Source Image", small_image);


//     big_image.convertTo(big_image, CV_32FC3);

//     std::string filename = "big_image.h5";
//     HDF5KernelPtr kernel(new HDF5Kernel(filename));
//     HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());

//     descriptions::HDF5IO hdf5io(kernel, schema);

//     hdf5io.ImageIO::save("image_group", "image_data", big_image);


//     cv::Mat big_image_loaded = *hdf5io.ImageIO::load("image_group", "image_data");

//     big_image_loaded.convertTo(big_image_loaded, src_type);

//     cv::Mat small_image_loaded;
//     cv::resize(big_image_loaded, small_image_loaded, cv::Size(1000, 500));
    
//     cv::imshow("small_image_loaded", small_image_loaded);
//     cv::waitKey(0);

// }

int main(int argc, char** argv)
{

    // directoryIOTest();
    debugTest();
    return 0;
    // directoryIOTest();
    // hdf5IOTest();
    // return 0;
    // // compressionTest();
    // // return 0;
    // // LOG.setLoggerLevel(Logger::DEBUG);
    // // debugTest();
    // // return 0;

    // LOG.setLoggerLevel(Logger::DEBUG);

    // LOG(lvr2::Logger::INFO) << "Directory IO" << std::endl;
    // directoryIOTest();
    // LOG(lvr2::Logger::HIGHLIGHT) << "Directory IO success" << std::endl;

    // LOG(lvr2::Logger::INFO) << "Hdf5 IO" << std::endl;
    // hdf5IOTest();
    // LOG(lvr2::Logger::HIGHLIGHT) << "Hdf5 IO success" << std::endl;


    // LOG.setLoggerLevel(Logger::DEBUG);
    // unitTest();
    // // loggerTest();
    // // std::cout << "\t" << "Bla" << Logger() << std::endl;

    // return 0;
    
    // if(argc > 1)
    // {
    //     std::string infilename = argv[1];
    //     std::cout << "Load file from '" << infilename << "' with old Hdf5 format." << std::endl;
    //     auto sp = loadOldHDF5(infilename);

    //     // std::cout << sp->positions[1]->transformation << std::endl;

    //     // YAML::Node node;
    //     // node = sp->positions[1]->transformation;
    //     // std::cout << node << std::endl;

    //     // Eigen::MatrixXd mat;
    //     // YAML::convert<Eigen::MatrixXd>::decode(node, mat);
    //     // std::cout << mat << std::endl;

    //     std::string outfilename = "scan_project.h5";
    //     HDF5KernelPtr kernel(new HDF5Kernel(outfilename));
    //     HDF5SchemaPtr schema(new ScanProjectSchemaRaw(outfilename));

    //     descriptions::HDF5IO hdf5io(kernel, schema);

    //     std::cout << "Write to '" << outfilename << "' with new Hdf5 format." << std::endl;
    //     hdf5io.save(sp);

        

    // }

    // // hdf5IOTest();
    // return 0;


    // metaHdf5Test();

    // return  0;

    // ioTest();
    // return 0;

    // hdf5_convert_old::Options options(argc, argv);

    // using OldIO = Hdf5Build<hdf5features::ScanProjectIO>;
    // using NewIO = descriptions::HDF5IO;

    // OldIO old_io;
    // old_io.open(argv[1]);

    // auto sp = old_io.ScanProjectIO::load();

    // if(sp)
    // {
    //     std::cout << "Loaded scan project!" << std::endl;
    //     std::string filename = "scan_project.h5";
        
    //     HDF5KernelPtr kernel(new HDF5Kernel(filename));
    //     HDF5SchemaPtr schema(new ScanProjectSchemaHDF5V2());
    //     NewIO io(kernel, schema);
    //     io.saveScanProject(sp);
    // } else {
    //     std::cout << "Could not load scan project" << std::endl;
    // }

    // ScanProjectPtr sp = dummyScanProject();

    // std::cout << "ScanProject with positions: " << sp->positions.size()  << std::endl;

    // std::string outdir = "test";
    // DirectoryKernelPtr kernel(new DirectoryKernel(outdir));
    // DirectorySchemaPtr schema(new ScanProjectSchemaRaw(outdir));
    // DirectoryIO io(kernel, schema);

    // std::cout << "SAVE" <<  std::endl;
    // io.saveScanProject(sp);

    // std::cout << "LOAD" << std::endl;
    // ScanProjectPtr sp_loaded =  io.loadScanProject();

    // std::cout << "COMPARE" << std::endl;
    // if(compare(sp, sp_loaded))
    // {
    //     std::cout << "success." << std::endl;
    // }  else {
    //     std::cout << "wrong." << std::endl;
    // }

    
    

    return 0;
}

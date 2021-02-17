#include "Logging.hpp"

#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSlam6D.hpp"
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

#include <boost/iostreams/code_converter.hpp>
#include <boost/iostreams/device/mapped_file.hpp>


#include "Hdf5ReaderOld.hpp"
#include "ScanTypesCompare.hpp"

using namespace lvr2;

PointBufferPtr operator*(Transformd T, PointBufferPtr points)
{
    PointBufferPtr ret(new PointBuffer);

    for(auto elem : *points)
    {
        (*ret)[elem.first] = elem.second;
    }

    if(ret->hasChannel<float>("points"))
    {
        Channel<float> points = ret->get<float>("points");
    
        for(size_t i=0; i<points.numElements(); i++)
        {
            Vector4d p(points[i][0], points[i][1], points[i][2], 1);
            Vector4d p_ = T * p;
            points[i][0] = p_(0);
            points[i][1] = p_(1);
            points[i][2] = p_(2);
        }

        ret->add<float>("points", points);
    }

    if(ret->hasChannel<float>("normals"))
    {
        Channel<float> normals = ret->get<float>("normals");

        for(size_t i=0; i<normals.numElements(); i++)
        {
            Vector3d n(normals[i][0], normals[i][1], normals[i][2]);
            Vector3d n_ = T.block<3,3>(0,0) * n;
            normals[i][0] = n_(0);
            normals[i][1] = n_(1);
            normals[i][2] = n_(2);
        }

        ret->add("normals", normals);
    }

    return ret;
}

ScanProjectPtr dummyScanProject()
{
    ScanProjectPtr ret(new ScanProject);

    for(size_t i=0; i<10; i++)
    {
        ScanPositionPtr scan_pos(new ScanPosition);
         
        scan_pos->transformation = Transformd::Identity();
        scan_pos->transformation(0,3) = static_cast<double>(i);
        scan_pos->poseEstimation = scan_pos->transformation;

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

                size_t npoints = scan->points->numPoints();
                Channel<float> normals(npoints, 3);

                for(size_t i=0; i<npoints; i++)
                {
                    normals[i][0] = 1.0;
                    normals[i][1] = 0.0;
                    normals[i][2] = 0.0;
                }

                scan->points->add("normals", normals);


                Transformd T = Transformd::Identity();
                T(2,3) = static_cast<double>(i);
                scan->points = T * scan->points;

                scan->numPoints = scan->points->numPoints();
                scan->startTime = 0.0;
                scan->endTime  = 100.0;

                lidar->scans.push_back(scan);
            }
            
            scan_pos->lidars.push_back(lidar);
        }

        for(size_t j=0; j<2; j++)
        {
            CameraPtr scan_cam(new Camera);
            scan_cam->transformation = Transformd::Identity();
            scan_cam->transformation(1,3) = -static_cast<double>(j);
            scan_cam->model.distortionModel = "opencv";
            scan_cam->model.k.resize(10);
            scan_cam->model.cx = 100.2;
            scan_cam->model.cy = 50.5;
            scan_cam->model.fx = 120.99;
            scan_cam->model.fy = 90.72;

            for(size_t k=0; k<10; k++)
            {
                scan_cam->model.k[k] = static_cast<double>(k) / 4.0;
            }
            
            for(size_t k=0; k<7; k++)
            {
                CameraImagePtr si = synthetic::genLVRImage();
                si->timestamp = 0.0;
                si->transformation = Transformd::Identity();
                si->transformation(2,3) = -static_cast<double>(k);
                si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(k + 1);
                scan_cam->images.push_back(si);
            }

            scan_cam->name = "Canon";
            scan_pos->cameras.push_back(scan_cam);
        }

        for(size_t j=0; j<2; j++)
        {
            HyperspectralCameraPtr h_cam(new HyperspectralCamera);

            h_cam->transformation = Transformd::Identity();
            h_cam->transformation(1,3) = -static_cast<double>(j);

            h_cam->model.principal(0) =  5.5;
            h_cam->model.principal(1) = 4.4;

            h_cam->model.focalLength(0) = 10.1;
            h_cam->model.focalLength(1) = 10.2;

            h_cam->model.distortion.resize(3);
            h_cam->model.distortion[0] = 2.0;
            h_cam->model.distortion[1] = 1.0;
            h_cam->model.distortion[2] = 0.5;

            for(size_t k=0; k<3; k++)
            {
                HyperspectralPanoramaPtr pano(new HyperspectralPanorama);

                pano->resolution[0] = 200;
                pano->resolution[1] = 200;
                pano->wavelength[0] = 100.0;
                pano->wavelength[1] = 900.25;

                for(size_t l=0; l<7; l++)
                {
                    HyperspectralPanoramaChannelPtr hchannel(new HyperspectralPanoramaChannel);

                    CameraImagePtr si = synthetic::genLVRImage();
                    hchannel->channel = si->image.clone();
                    hchannel->timestamp = 0.0;
                    pano->channels.push_back(hchannel);
                }

                h_cam->panoramas.push_back(pano);
            }

            scan_pos->hyperspectral_cameras.push_back(h_cam);
        }

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

    return equal(sp, sp_loaded);
}

bool hdf5IOTest()
{
    std::string filename = "scan_project.h5";
    HDF5KernelPtr kernel(new HDF5Kernel(filename));
    HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());

    descriptions::HDF5IO hdf5io(kernel, schema);

    auto sp = dummyScanProject();
    hdf5io.save(sp);

    auto sp_loaded = hdf5io.ScanProjectIO::load();
    return equal(sp, sp_loaded);
}

struct Slam6DConfig 
{
    std::string executable = "/home/amock/software/3DTK/bin/slam6D";
    std::string tmpdir = "tmp_slam";

    int start = 0;
    int end = -1;
    
    /**
     * @brief Minimization method for ICP
     * 1 = unit quaternion based method by Horn
     * 2 = singular value decomposition by Arun et al.
     * 3 = orthonormal matrices by Horn et al.
     * 4 = dual quaternion method by Walker et al.
     * 5 = helix approximation by Hofer & Potmann
     * 6 = small angle approximation
     * 7 = Lu & Milios style, i.e., uncertainty based, with Euler angles
     * 8 = Lu & Milios style, i.e., uncertainty based, with Quaternion
     * 9 = unit quaternion with scale method by Horn
     */
    size_t algo = 1;

    /**
     * @brief selects the Nearest Neighbor Search Algorithm
     * 0 = simple k-d tree
     * 1 = cached k-d tree
     * 2 = ANNTree
     * 3 = BOCTree
     */
    size_t nns_method = 0;

    /**
     * @brief selects the method for closing the loop explicitly
     * 0 = no loop closing technique
     * 1 = euler angles
     * 2 = quaternions
     * 3 = unit quaternions
     * 4 = SLERP (recommended)
     */
    size_t loop6DAlgo = 0;

    /**
     * @brief selects the minimizazion method for the SLAM matching algorithm
     * 0 = no global relaxation technique
     * 1 = Lu & Milios extension using euler angles due to Borrmann et al.
     * 2 = Lu & Milios extension using using unit quaternions
     * 3 = HELIX approximation by Hofer and Pottmann
     * 4 = small angle approximation
     */
    size_t graphSlam6DAlgo = 0;

    /**
     * @brief sets the maximal number of ICP iterations to <NR>
     */
    int iter = 50;

    /**
     * @brief sets the maximal number of iterations 
     * for SLAM to <NR>(if not set, graphSLAM is not executed)
     */
    int iterSLAM = -1;

    /**
     * @brief neglegt all data points with a distance larger than NR 'units'
     * 
     */
    double max = -1.0; 

    double epsICP = 0.00001;
    double epsSLAM = 0.5; // careful (cm)
};

ScanProjectPtr slam6d(ScanProjectPtr sp, Slam6DConfig config)
{
    // save to slam6d structure
    DirectoryKernelPtr kernel(new DirectoryKernel(config.tmpdir));
    DirectorySchemaPtr schema(new ScanProjectSchemaSlam6D(config.tmpdir));
    DirectoryIO dirio(kernel, schema);
    dirio.save(sp);

    std::stringstream ss;
    ss << config.executable;
    
    // add parameters here
    
    ss << " -s " << config.start;
    if(config.end >= 0)
    {
        ss << " -e " << config.end;
    }
    
    ss << " -a " << config.algo;
    ss << " -t " << config.nns_method;
    ss << " -L " << config.loop6DAlgo;
    ss << " -G " << config.graphSlam6DAlgo;
    ss << " -i " << config.iter;
    if(config.iterSLAM >= 0)
    {
        ss << " -I " << config.iterSLAM;
    }
    if(config.max >= 0.0)
    {
        ss << " -m " << config.max;
    }
    ss << " -5 " << config.epsICP;
    ss << " -6 " << config.epsSLAM;

    ss << " " << config.tmpdir;
    system(ss.str().c_str());

    // load slam6d
    return dirio.ScanProjectIO::load();
}

void registrationTest()
{
    auto sp = dummyScanProject();

    Slam6DConfig config;
    auto sp_registered = slam6d(sp, config);

    for(size_t i=0; i<sp_registered->positions.size(); i++)
    {
        std::cout << std::endl;
        std::cout << "Position " << i <<  " Corrected estimation " << std::endl;
        std::cout << sp->positions[i]->poseEstimation << std::endl;
        std::cout << "To" << std::endl;
        std::cout << sp_registered->positions[i]->transformation << std::endl;
    }
}

bool slam6dIOTest()
{
    std::string dirname = "slam6d_directory";

    DirectoryKernelPtr kernel2(new DirectoryKernel(dirname));
    DirectorySchemaPtr schema2(new ScanProjectSchemaSlam6D(dirname));
    DirectoryIO dirio(kernel2, schema2);

    LOG(lvr2::Logger::DEBUG) << "Create Dummy Scanproject..." << std::endl;
    auto sp = dummyScanProject();

    // register 

    LOG(lvr2::Logger::DEBUG) << "Save Scanproject to directory..." << std::endl;
    dirio.save(sp);

    // do slam6d
    std::string slam6dfile = "/home/amock/software/3DTK/bin/slam6D";


    std::stringstream ss;
    ss << slam6dfile;
    ss << " ";
    ss << dirname;
    system(ss.str().c_str());

    // return true;
    LOG(lvr2::Logger::DEBUG) << "Load Scanproject from directory..." << std::endl;
    auto sp_loaded = dirio.ScanProjectIO::load();


    for(size_t i=0; i<sp_loaded->positions.size(); i++)
    {
        std::cout << std::endl;
        std::cout << "Position " << i <<  " Corrected estimation " << std::endl;
        std::cout << sp->positions[i]->poseEstimation << std::endl;
        std::cout << "To" << std::endl;
        std::cout << sp_loaded->positions[i]->transformation << std::endl;
    }



    return equal(sp, sp_loaded);
}

void unitTest()
{
    LOG(Logger::INFO) << "Hdf5-IO Test" << std::endl;
    
    LOG.tab();
    bool hdf5_success = true;
    for(size_t i=0; i<10; i++)
    {
        if(!hdf5IOTest()) {
            hdf5_success = false;
            break;
        }
        LOG(Logger::DEBUG) << i << " success." << std::endl;
    }
    LOG.deltab();
    if(hdf5_success)
    {
        LOG(Logger::HIGHLIGHT) << "Hdf5-IO success." << std::endl;
    } else {
        LOG(Logger::ERROR) << "TODO: Fix Hdf5-IO" << std::endl;
    }
    
    std::cout << std::endl;
    LOG(Logger::INFO) << "Directory-IO Test" << std::endl;
    LOG.tab();
    bool dir_success = true;
    for(size_t i=0; i<10; i++)
    {
        if(!directoryIOTest()) {
            dir_success = false;
            break;
        }
        LOG(Logger::DEBUG) << i << " success." << std::endl;
    }
    LOG.deltab();

    if(dir_success) {
        LOG(Logger::HIGHLIGHT) << "Directory-IO success." << std::endl;
    } else {
        LOG(Logger::ERROR) << "TODO: Fix Directory-IO" << std::endl;
    }
}

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

    ScanPtr scan(new Scan);
    scan->points = points;
    
    dirio.ScanIO::save(posNo, lidarNo, scanNo, scan);

    dirio.ScanIO::load(posNo, lidarNo, scanNo);

    // dirio.PointCloudIO::save(posNo, lidarNo, scanNo, points);
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

void metaOnlyTest()
{
    std::string dirname = "meta_io_test";

    DirectoryKernelPtr kernel(new DirectoryKernel(dirname));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(dirname));
    DirectoryIO dirio(kernel, schema);

    auto sp = dummyScanProject();

    dirio.save(sp);
    

}

namespace bio = boost::iostreams;

void writeBigFile()
{
    std::string filename = "/home/amock/datasets/hello.data";

    // 300 millionen punkte
    size_t Npoints = 300000000;

    doubleArr points(new double[Npoints * 3]);
    for(size_t i=0; i<Npoints * 3; i++)
    {
        points[i] = i;
    }

    std::vector<size_t> shape = {Npoints, 3};
    std::cout << "Save!" << std::endl;
    dataIOsave(filename, shape, points);
    std::cout << "Finished." << std::endl;
}

void memoryMapTest()
{
    std::string filename = "/home/amock/datasets/hello.data";


    DataIOHeader header = dataIOloadHeader("hello.data");

    // std::cout << "Loaded Header" << std::endl;
    // std::cout << header << std::endl;


    // bio::mapped_file_params params;
    // params.path = "hello.data";
    // // params.new_file_size = std::pow(1024,2);
    // params.flags = bio::mapped_file::mapmode::readonly;
    // bio::mapped_file mf;
    // mf.open(params);

    // size_t offset = sizeof(DataIOHeader) + header.JSON_BYTES;
    // std::cout << "MMap at " << offset << std::endl;
    
    // // char* begin = mf.data();

    // char* bytes = (char*)mf.const_data();
    // char* begin = bytes + offset;

    // double* data = reinterpret_cast<double*>(begin);

    // std::cout << data[10] << std::endl;

    // // std::cout << data[0] << data[1] << bytes[2] << bytes[3] << std::endl;

    // // std::cout << *reinterpret_cast<double*>(begin) << std::endl;

    // // // char* bytes = begin;
    // // // for (size_t i = 0; i < 10; ++i)
    // // //     bytes[i] = 'C';

    // mf.close();

}

int main(int argc, char** argv)
{
    LOG.setLoggerLevel(lvr2::Logger::DEBUG);

    writeBigFile();
    // memoryMapTest();
    return 0;

    // return 0;
    
    
    // YAML::Node node;
    // node["bla"] = "bla";

    // std::cout << node.Type() << std::endl;

    // std::cout << YAML::NodeType::Undefined << std::endl;
    // std::cout << YAML::NodeType::Null << std::endl;
    // std::cout << YAML::NodeType::Scalar << std::endl;
    // std::cout << YAML::NodeType::Sequence << std::endl;
    // std::cout << YAML::NodeType::Map << std::endl;
    
    
    // hdf5IOTest();
    directoryIOTest();
    // debugTest();
    // return 0;
    // directoryIOTest();
    // slam6dIOTest();
    // registrationTest();
    metaOnlyTest();
    // hdf5IOTest();
    return 0;
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


    LOG.setLoggerLevel(Logger::DEBUG);
    unitTest();
    // // loggerTest();
    // // std::cout << "\t" << "Bla" << Logger() << std::endl;

    return 0;
    
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

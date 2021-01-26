#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
// #include "lvr2/io/descriptions/HDF5IO.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"


#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

// #include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

// #include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
// #include "lvr2/io/hdf5/ScanProjectIO.hpp"

#include <boost/filesystem.hpp>

#include "lvr2/util/Synthetic.hpp"

#include <boost/type_index.hpp>

using namespace lvr2;

ScanProjectPtr dummyScanProject()
{
    ScanProjectPtr ret(new ScanProject);

    for(size_t i=0; i<5; i++)
    {
        ScanPositionPtr scan_pos(new ScanPosition);

        for(size_t j=0; j<2; j++)
        {
            LIDARPtr lidar(new LIDAR);
            lidar->name = "Riegl";

            for(size_t k=0; k<10; k++)
            {
                ScanPtr scan(new Scan);

                scan->hResolution = 1.0;
                scan->vResolution = 0.2;
                scan->thetaMin = -M_PI;
                scan->thetaMax = M_PI;
                // scan->poseEstimation = lvr2::Transformd::Identity();

                scan->registration = lvr2::Transformd::Identity();

                scan->points = synthetic::genSpherePoints(50, 50);

                scan->numPoints = scan->points->numPoints();
                scan->startTime = 0.0;
                scan->endTime  = 100.0;
                scan->poseEstimation = Transformd::Identity() * 2;

                lidar->scans.push_back(scan);
            }

            
            scan_pos->lidars.push_back(lidar);
        }

        for(size_t j=0; j<2; j++)
        {
            CameraPtr scan_cam(new Camera);
            scan_cam->model.distortionModel = "opencv";
            scan_cam->model.k.resize(10);
            for(size_t k=0; k<10; k++)
            {
                scan_cam->model.k[k] = static_cast<double>(k);
            }
            
            for(size_t k=0; k<10; k++)
            {
                CameraImagePtr si = synthetic::genLVRImage();
                si->extrinsics = Extrinsicsd::Identity() * static_cast<double>(k);
                si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(k + 1);
                scan_cam->images.push_back(si);
            }

            scan_cam->name = "Canon";

            scan_pos->cameras.push_back(scan_cam);

        }

        ret->positions.push_back(scan_pos);
    }

    ret->unit = "m";
    ret->coordinateSystem = "right-handed";
    ret->transformation = Transformd::Identity();

    return ret;
}

// bool compare(CameraImagePtr si1, CameraImagePtr si2)
// {
//     if(!si1 && si2){return false;}
//     if(si1 && !si2){return false;}
//     if(!si1 && !si2){return true;}

//     if(!si1->extrinsics.isApprox(si2->extrinsics)){ 
//         std::cout << "ScanImage extriniscs differ: " << std::endl;
//         std::cout << si1->extrinsics << std::endl;
//         std::cout << " != " << std::endl;
//         std::cout << si2->extrinsics << std::endl;
//         return false; 
//     }

//     if(!si1->extrinsicsEstimate.isApprox(si2->extrinsicsEstimate)){ 
//         std::cout << "ScanImage extriniscsEstimate differ: " << std::endl;
//         std::cout << si1->extrinsicsEstimate << std::endl;
//         std::cout << " != " << std::endl;
//         std::cout << si2->extrinsicsEstimate << std::endl;
//         return false; 
//     }

//     if(si1->imageFile != si2->imageFile)
//     {
//         std::cout << "ScanImage imageFile differ: " << si1->imageFile << " != " << si2->imageFile << std::endl;
//         return false;
//     }

//     // if(cv::countNonZero(si1->image != si2->image) != 0){
//     //     std::cout <<"ScanImage image data differ "  << std::endl;
//     //     cv::imshow("ScanImage 1", si1->image);
//     //     cv::imshow("ScanImage 2", si2->image);
//     //     cv::waitKey(0);
//     //     return false;
//     // }

//     return true;
// }

bool compare(CameraPtr sc1, CameraPtr sc2)
{
    if(!sc1 && sc2){return false;}
    if(sc1 && !sc2){return false;}
    if(!sc1 && !sc2){return true;}

    if(sc1->model.cx != sc2->model.cx)
    {
        std::cout << "ScanCamera cx differ: " << sc1->model.cx << " != " << sc2->model.cx << std::endl;
        return false;
    }

    if(sc1->model.cy != sc2->model.cy)
    {
        std::cout << "ScanCamera cy differ: "  << sc1->model.cy << " != " << sc2->model.cy << std::endl;
        return false;
    }

    if(sc1->model.fx != sc2->model.fx)
    {
        std::cout << "ScanCamera fx differ: " << sc1->model.fx << " != " << sc2->model.fx <<  std::endl;
        return false;
    }

    if(sc1->model.fy != sc2->model.fy)
    {
        std::cout << "ScanCamera fy differ: " << sc1->model.fy << " != " << sc2->model.fy << std::endl;
        return false;
    }

    if(sc1->name != sc2->name)
    {
        std::cout << "ScanCamera SensorName differ: " << sc1->name << " != " << sc2->name << std::endl;
        return false;
    }

    if(sc1->images.size() != sc2->images.size())
    {
        std::cout << "ScanCamera number of images differ: " << sc1->images.size() << " != "  << sc2->images.size() << std::endl;
        return false;
    }

    // for(size_t i=0; i<sc1->images.size(); i++)
    // {
    //     if(!compare(sc1->images[i], sc2->images[i]))
    //     {
    //         return false;
    //     }
    // }

    return true;
}

bool compare(PointBufferPtr p1, PointBufferPtr p2)
{
    if(!p1 && p2){
        std::cout << "P1 is empty but not P2" << std::endl;
        return false;}
    if(p1 && !p2){
        std::cout << "P2 is empty but not P1" << std::endl;
        return false;}
    if(!p1 && !p2){return true;}

    for(auto elem : *p1)
    {
        auto it = p2->find(elem.first);
        if(it != p2->end())
        {
            // found channel in second pointbuffer
            if(elem.second.type() != it->second.type())
            {
                std::cout << "Type differ for " << elem.first << std::endl;
                return false;
            }

            if(elem.second.numElements() != it->second.numElements())
            {
                std::cout << "numElements differ for " << elem.first << std::endl;
                return false;
            }

            if(elem.second.width() != it->second.width())
            {
                std::cout << "width differ for " << elem.first << std::endl;
                return false;
            }

        } else {
            std::cout << "Could not find channel " << elem.first << " in p2" << std::endl;
            return false;
        }
    }

    return true;
}

// bool compare(ScanPtr s1, ScanPtr s2)
// {
//     if(!s1 && s2){return false;}
//     if(s1 && !s2){return false;}
//     if(!s1 && !s2){return true;}
    
//     if(s1->positionNumber != s2->positionNumber){
//         std::cout << "Scan: positionNumber differs: " << s1->positionNumber << " <-> " << s2->positionNumber << std::endl;
//         return false;}
//     if(s1->startTime != s2->startTime){
//         std::cout << "Scan: startTime differs: " << s1->startTime << " <-> " << s2->startTime << std::endl;
//         return false;}
//     if(s1->endTime != s2->endTime){
//         std::cout << "Scan: endTime differs: " << s1->endTime << " <-> " << s2->endTime << std::endl;
//         return false;}
//     if(s1->numPoints != s2->numPoints){
//         std::cout << "Scan: numPoints differs: " << s1->numPoints << " <-> " << s2->numPoints << std::endl;
//         return false;}
//     if(s1->phiMin != s2->phiMin){
//         std::cout << "Scan: phiMin differs: " << s1->phiMin << " <-> " << s2->phiMin << std::endl;
//         return false;}
//     if(s1->phiMax != s2->phiMax){
//         std::cout << "Scan: phiMax differs: " << s1->phiMax << " <-> " << s2->phiMax << std::endl;
//         return false;}
//     if(s1->thetaMin != s2->thetaMin){
//         std::cout << "Scan: thetaMin differs: " << s1->thetaMin << " <-> " << s2->thetaMin << std::endl;
//         return false;}
//     if(s1->thetaMax != s2->thetaMax){
//         std::cout << "Scan: thetaMax differs: " << s1->thetaMax << " <-> " << s2->thetaMax << std::endl;
//         return false;}
    
//     if(s1->vResolution != s2->vResolution){
//         std::cout << "Scan: vResolution differs: " << s1->vResolution << " <-> " << s2->vResolution << std::endl;
//         return false;}
    
//     if(s1->hResolution != s2->hResolution){
//         std::cout << "Scan: hResolution differs: " << s1->hResolution << " <-> " << s2->hResolution << std::endl;
//         return false;}

//     if(!compare(s1->points, s2->points)){
//         std::cout << "scan points differ" << std::endl;
//         return false;}

//     return true;
// }

bool compare(ScanPositionPtr sp1, ScanPositionPtr sp2)
{
    if(!sp1 && sp2){return false;}
    if(sp1 && !sp2){return false;}
    if(!sp1 && !sp2){return true;}

    // gps
    // if(sp1->longitude != sp2->longitude){ return false; }
    // if(sp1->latitude != sp2->latitude){ return false; }
    // if(sp1->altitude != sp2->altitude){ return false; }
    // timestamp
    if(sp1->timestamp != sp2->timestamp){ 
        std::cout <<  "ScanPosition timestamp differs" << std::endl;
        return false; 
    }
    // pose
    if(!sp1->pose_estimation.isApprox(sp2->pose_estimation)){ 
        std::cout <<  "ScanPosition pose_estimation differs" << std::endl;
        return false; 
    }
    if(!sp1->transformation.isApprox(sp2->transformation)){ 
        std::cout <<  "ScanPosition transformation differs" << std::endl;
        return false; 
    }

    // vectors
    // if(sp1->scans.size() != sp2->scans.size()){return false;}
    if(sp1->cameras.size() != sp2->cameras.size()){
        std::cout <<  "ScanPosition Ncameras differs: " << sp1->cameras.size()  << " != " << sp2->cameras.size() << std::endl;
        return false;
    }

    // scans
    // for(size_t i=0; i<sp1->scans.size(); i++)
    // {
    //     if(!compare(sp1->scans[i], sp2->scans[i]))
    //     {
    //         return false;
    //     }
    // }

    // cams
    for(size_t i=0; i<sp1->cameras.size(); i++)
    {
        if(!compare(sp1->cameras[i], sp2->cameras[i]))
        {
            return false;
        }
    }

    return true;
}

bool compare(ScanProjectPtr sp1, ScanProjectPtr sp2)
{
    // compare shared ptr
    if(!sp1 && sp2){return false;}
    if(sp1 && !sp2){return false;}
    if(!sp1 && !sp2){return true;}

    if(sp1->coordinateSystem != sp2->coordinateSystem)
    {
        std::cout << "ScanProject coordinateSystem differ"  << std::endl;
        return false;
    }

    if(!sp1->transformation.isApprox(sp2->transformation) )
    {
        std::cout << "ScanProject transformation differ"  << std::endl;
        return false;
    }

    if(sp1->name != sp2->name)
    {
        std::cout << "ScanProject name differ"  << std::endl;
        return false;
    }

    if(sp1->positions.size() != sp2->positions.size())
    {
        std::cout << "ScanProject Npositions differ"  << std::endl;
        return false;
    }

    for(size_t i=0; i<sp1->positions.size(); i++)
    {
        if(!compare(sp1->positions[i], sp2->positions[i]))
        {
            return false;
        }
    }

    return true;
}

void ioTest()
{
    // std::string filename = "test";

    // ScanProjectPtr sp = dummyScanProject();

    // /// WRITE TO HDF5
    // HDF5KernelPtr hdf5_kernel(new HDF5Kernel(filename + ".h5"));
    // HDF5SchemaPtr hdf5_schema(new ScanProjectSchemaHDF5V2());
    // descriptions::HDF5IO hdf5_io(hdf5_kernel, hdf5_schema);
    // std::cout << "--------------------------" << std::endl;
    // std::cout << "SAVE SCANPROJECT" << std::endl;
    // hdf5_io.saveScanProject(sp);

    // std::cout << "--------------------------" << std::endl;
    // std::cout << "LOAD SCANPROJECT" << std::endl;
    // ScanProjectPtr sp_loaded = hdf5_io.loadScanProject();

    // if(compare(sp, sp_loaded))
    // {
    //     std::cout << "IO works correct!" << std::endl;
    // } else {
    //     std::cout << "Something is wrong!" << std::endl;
    // }
}

int main(int argc, char** argv)
{
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

    ScanProjectPtr sp = dummyScanProject();

    std::cout << "ScanProject with positions: " << sp->positions.size()  << std::endl;

    std::string outdir = "test";
    DirectoryKernelPtr kernel(new DirectoryKernel(outdir));
    DirectorySchemaPtr schema(new ScanProjectSchemaRaw(outdir));
    DirectoryIO io(kernel, schema);

    std::cout << "SAVE" <<  std::endl;
    io.saveScanProject(sp);

    std::cout << "LOAD" << std::endl;
    ScanProjectPtr sp_loaded =  io.loadScanProject();

    std::cout << "COMPARE" << std::endl;
    if(compare(sp, sp_loaded))
    {
        std::cout << "success." << std::endl;
    }  else {
        std::cout << "wrong." << std::endl;
    }



    
    return 0;
}

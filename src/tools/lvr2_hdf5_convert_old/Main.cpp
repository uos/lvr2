#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"


#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ScanProjectIO.hpp"

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
            ScanPtr scan(new Scan);


            scan->hResolution = 1.0;
            scan->vResolution = 0.2;
            scan->thetaMin = -M_PI;
            scan->thetaMax = M_PI;
            scan->poseEstimation = lvr2::Transformd::Identity();

            scan->registration = lvr2::Transformd::Identity();

            scan->points = synthetic::genSpherePoints(50, 50);

            scan_pos->scans.push_back(scan);

            scan->numPoints = scan->points->numPoints();
        }

        for(size_t j=0; j<2; j++)
        {
            ScanCameraPtr scan_cam(new ScanCamera);
            scan_cam->camera.distortionModel = "opencv";
            for(size_t k=0; k<10; k++)
            {
                scan_cam->images.push_back(synthetic::genLVRImage());
            }
            scan_pos->cams.push_back(scan_cam);
        }

        ret->positions.push_back(scan_pos);
    }

    ret->coordinateSystem = "righthanded";
    ret->pose = Transformd::Identity();

    return ret;
}

bool compare(ScanCameraPtr sc1, ScanCameraPtr sc2)
{
    if(!sc1 && sc2){return false;}
    if(sc1 && !sc2){return false;}
    if(!sc1 && !sc2){return true;}

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

bool compare(ScanPtr s1, ScanPtr s2)
{
    if(!s1 && s2){return false;}
    if(s1 && !s2){return false;}
    if(!s1 && !s2){return true;}
    
    if(s1->positionNumber != s2->positionNumber){
        std::cout << "Scan: positionNumber differs: " << s1->positionNumber << " <-> " << s2->positionNumber << std::endl;
        return false;}
    if(s1->startTime != s2->startTime){
        std::cout << "Scan: startTime differs: " << s1->startTime << " <-> " << s2->startTime << std::endl;
        return false;}
    if(s1->endTime != s2->endTime){
        std::cout << "Scan: endTime differs: " << s1->endTime << " <-> " << s2->endTime << std::endl;
        return false;}
    if(s1->numPoints != s2->numPoints){
        std::cout << "Scan: numPoints differs: " << s1->numPoints << " <-> " << s2->numPoints << std::endl;
        return false;}
    if(s1->phiMin != s2->phiMin){
        std::cout << "Scan: phiMin differs: " << s1->phiMin << " <-> " << s2->phiMin << std::endl;
        return false;}
    if(s1->phiMax != s2->phiMax){
        std::cout << "Scan: phiMax differs: " << s1->phiMax << " <-> " << s2->phiMax << std::endl;
        return false;}
    if(s1->thetaMin != s2->thetaMin){
        std::cout << "Scan: thetaMin differs: " << s1->thetaMin << " <-> " << s2->thetaMin << std::endl;
        return false;}
    if(s1->thetaMax != s2->thetaMax){
        std::cout << "Scan: thetaMax differs: " << s1->thetaMax << " <-> " << s2->thetaMax << std::endl;
        return false;}
    
    if(s1->vResolution != s2->vResolution){
        std::cout << "Scan: vResolution differs: " << s1->vResolution << " <-> " << s2->vResolution << std::endl;
        return false;}
    
    if(s1->hResolution != s2->hResolution){
        std::cout << "Scan: hResolution differs: " << s1->hResolution << " <-> " << s2->hResolution << std::endl;
        return false;}

    if(!compare(s1->points, s2->points)){
        std::cout << "scan points differ" << std::endl;
        return false;}

    return true;
}

bool compare(ScanPositionPtr sp1, ScanPositionPtr sp2)
{
    if(!sp1 && sp2){return false;}
    if(sp1 && !sp2){return false;}
    if(!sp1 && !sp2){return true;}

    // gps
    if(sp1->longitude != sp2->longitude){ return false; }
    if(sp1->latitude != sp2->latitude){ return false; }
    if(sp1->altitude != sp2->altitude){ return false; }
    // timestamp
    if(sp1->timestamp != sp2->timestamp){ return false; }
    // pose
    if(!sp1->pose_estimate.isApprox(sp2->pose_estimate)){ return false; }
    if(!sp1->registration.isApprox(sp2->registration)){ return false; }

    // vectors
    if(sp1->scans.size() != sp2->scans.size()){return false;}
    if(sp1->cams.size() != sp2->cams.size()){return false;}

    // scans
    for(size_t i=0; i<sp1->scans.size(); i++)
    {
        if(!compare(sp1->scans[i], sp2->scans[i]))
        {
            return false;
        }
    }

    // cams
    for(size_t i=0; i<sp1->cams.size(); i++)
    {
        if(!compare(sp1->cams[i], sp2->cams[i]))
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
        return false;
    }

    if(!sp1->pose.isApprox(sp2->pose) )
    {
        return false;
    }

    if(sp1->sensorName != sp2->sensorName)
    {
        return false;
    }

    if(sp1->positions.size() != sp2->positions.size())
    {
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

void writeTest()
{
    std::string filename = "test";

    ScanProjectPtr sp = dummyScanProject();

    /// WRITE TO HDF5
    HDF5KernelPtr hdf5_kernel(new HDF5Kernel(filename + ".h5"));
    HDF5SchemaPtr hdf5_schema(new ScanProjectSchemaHDF5V2());
    descriptions::HDF5IO hdf5_io(hdf5_kernel, hdf5_schema);
    hdf5_io.saveScanProject(sp);

    std::cout << "--------------------------" << std::endl;
    std::cout << "LOAD SCANPROJECT" << std::endl;
    ScanProjectPtr sp_loaded = hdf5_io.loadScanProject();

    if(compare(sp, sp_loaded))
    {
        std::cout << "IO works correct!" << std::endl;
    } else {
        std::cout << "Something is wrong!" << std::endl;
    }
}

void loadTest()
{
    std::string filename = "test";

    ScanProjectPtr sp = dummyScanProject();


    /// WRITE TO HDF5

    HDF5KernelPtr hdf5_kernel(new HDF5Kernel(filename + ".h5"));
    HDF5SchemaPtr hdf5_schema(new ScanProjectSchemaHDF5V2());
    descriptions::HDF5IO hdf5_io(hdf5_kernel, hdf5_schema);
    // hdf5_io.saveScanProject(sp);

    std::cout << "--------------------------" << std::endl;
    std::cout << "LOAD SCANPROJECT" << std::endl;
    ScanProjectPtr sp_loaded = hdf5_io.loadScanProject();

    // DirectoryKernelPtr dir_kernel(new DirectoryKernel(filename));
    // DirectorySchemaPtr dir_schema(new ScanProjectSchemaRaw(filename));
    // DirectoryIO dir_io(dir_kernel, dir_schema);
    // dir_io.saveScanProject(sp);
}

int main(int argc, char** argv)
{
    hdf5_convert_old::Options options(argc, argv);

    using OldIO = Hdf5Build<hdf5features::ScanProjectIO>;
    using NewIO = descriptions::HDF5IO;

    OldIO old_io;
    old_io.open(argv[1]);

    auto sp = old_io.ScanProjectIO::load();

    if(sp)
    {
        std::cout << "Loaded scan project!" << std::endl;
        std::string filename = "scan_project.h5";
        
        HDF5KernelPtr kernel(new HDF5Kernel(filename));
        HDF5SchemaPtr schema(new ScanProjectSchemaHDF5V2());
        NewIO io(kernel, schema);
        io.saveScanProject(sp);
    } else {
        std::cout << "Could not load scan project" << std::endl;
    }

    return 0;
}

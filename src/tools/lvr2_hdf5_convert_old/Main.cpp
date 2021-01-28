#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"
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
                scan->poseEstimation = lvr2::Transformd::Identity();
                scan->transformation = lvr2::Transformd::Identity();

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

template<typename HighFiveContainerT>
void writeMeta(HighFiveContainerT g, YAML::Node meta, std::string prefix = "")
{
    
    for(YAML::const_iterator it=meta.begin(); it != meta.end(); ++it) 
    {   
        std::string key = it->first.as<std::string>();
        YAML::Node value = it->second;

        // attributeName of hdf5
        std::string attributeName = key;
        
        // add prefix to key
        if(prefix != ""){ attributeName = prefix + "/" + attributeName; }

        if(value.Type() == YAML::NodeType::Scalar)
        {
            // Write Scalar
            // std::cout << attributeName << ": Scalar" << std::endl;

            // get scalar type
            long int lint;
            double dbl;
            bool bl;
            std::string str;

            if(YAML::convert<long int>::decode(value, lint))
            {
                hdf5util::setAttribute(g, attributeName, lint);
            } 
            else if(YAML::convert<double>::decode(value, dbl)) 
            {
                hdf5util::setAttribute(g, attributeName, dbl);
            } 
            else if(YAML::convert<bool>::decode(value, bl))
            {
                hdf5util::setAttribute(g, attributeName, bl);
            } 
            else if(YAML::convert<std::string>::decode(value, str))
            {
                hdf5util::setAttribute(g, attributeName, str);
            }
            else
            {
                std::cout << "ERROR: UNKNOWN TYPE of value " << value << std::endl;
            }
        } 
        else if(value.Type() == YAML::NodeType::Sequence) 
        {
            // check the type with all elements
            bool is_int = true;
            bool is_double = true;
            bool is_bool = true;
            size_t nelements = 0;

            for(auto it = value.begin(); it != value.end(); it++)
            {
                long int lint;
                double dbl;
                bool bl;
                if(!YAML::convert<long int>::decode(*it, lint))
                {
                    is_int = false;
                }

                if(!YAML::convert<double>::decode(*it, dbl))
                {
                    is_double = false;
                }

                if(!YAML::convert<bool>::decode(*it, bl))
                {
                    is_bool = false;
                }

                nelements++;
            }

            if(is_int)
            {
                std::vector<long int> data;
                for(auto it = value.begin(); it != value.end(); it++)
                {
                    data.push_back(it->as<long int>());
                }
                hdf5util::setAttributeVector(g, attributeName, data);
            }
            else if(is_double)
            {
                std::vector<double> data;
                for(auto it = value.begin(); it != value.end(); it++)
                {
                    data.push_back(it->as<double>());
                }
                hdf5util::setAttributeVector(g, attributeName, data);
            }
            else if(is_bool)
            {
                // Bool vector is special
                // https://stackoverflow.com/questions/51352045/void-value-not-ignored-as-it-ought-to-be-on-non-void-function
                // need workaround

                // hdf5 stores bool arrays in uint8 anyway
                // std::vector<uint8_t> data;
                // for(auto it = value.begin(); it != value.end(); it++)
                // {
                //     data.push_back(static_cast<uint8_t>(it->as<bool>()));
                // }
                // hdf5util::setAttributeVector(g, attributeName, data);

                boost::shared_array<bool> data(new bool[nelements]);
                size_t i = 0;
                for(auto it = value.begin(); it != value.end(); it++, i++)
                {
                    data[i] = it->as<bool>();
                }
                hdf5util::setAttributeArray(g, attributeName, data, nelements);

            } else {
                std::cout << "Tried to write YAML list of unknown typed elements: " << *it << std::endl;
            }
        } 
        else if(value.Type() == YAML::NodeType::Map) 
        {
            // check if Map is known type
            if(YAML::isMatrix(value))
            {
                Eigen::MatrixXd mat;
                if(YAML::convert<Eigen::MatrixXd>::decode(value, mat))
                {
                    hdf5util::setAttributeMatrix(g, attributeName, mat);
                } else {
                    std::cout << "ERROR matrix" << std::endl;
                }
            } else {
                writeMeta(g, value, attributeName);
            }
        } 
        else 
        {
            std::cout << attributeName << ": UNKNOWN -> Error" << std::endl;
            std::cout << value << std::endl;
        }
    }
}

template<typename HighFiveContainerT>
YAML::Node readMeta(HighFiveContainerT g)
{
    YAML::Node ret = YAML::Load("");

    for(std::string attributeName : g.listAttributeNames())
    {
        std::vector<YAML::Node> yamlNodes;
        std::vector<std::string> yamlNames = hdf5util::splitGroupNames(attributeName);

        auto node_iter = ret;
        yamlNodes.push_back(node_iter);
        for(size_t i=0; i<yamlNames.size()-1; i++)
        {
            YAML::Node tmp = yamlNodes[i][yamlNames[i]];
            yamlNodes.push_back(tmp);
        }

        YAML::Node back = yamlNodes.back();

        HighFive::Attribute h5attr = g.getAttribute(attributeName);
        std::vector<size_t> dims = h5attr.getSpace().getDimensions();
        HighFive::DataType h5type = h5attr.getDataType();
        if(dims.size() == 0)
        {
            // Bool problems
            if(h5type == HighFive::AtomicType<bool>())
            {
                back[yamlNames.back()] = *hdf5util::getAttribute<bool>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<char>())
            {
                back[yamlNames.back()] = *hdf5util::getAttribute<char>(g, attributeName);
            } 
            else if(h5type == HighFive::AtomicType<unsigned char>())
            {
                back[yamlNames.back()] = *hdf5util::getAttribute<unsigned char>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<short>())
            {
                back[yamlNames.back()] = *hdf5util::getAttribute<short>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned short>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<unsigned short>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<int>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned int>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<unsigned int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<long int>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<long int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned long int>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<unsigned long int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<float>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<float>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<double>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<double>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<bool>())
            {   
                back[yamlNames.back()] = *hdf5util::getAttribute<bool>(g, attributeName);
            } 
            else if(h5type == HighFive::AtomicType<std::string>()) 
            {
                back[yamlNames.back()] = *hdf5util::getAttribute<std::string>(g, attributeName);
            } 
            else {
                std::cout << h5type.string() << ": type not implemented. " << std::endl;
            }
        }
        else if(dims.size() == 1)
        {
            back[yamlNames.back()] = YAML::Load("[]");
            // Sequence
            if(h5type == HighFive::AtomicType<bool>())
            {
                std::vector<uint8_t> data = *hdf5util::getAttributeVector<uint8_t>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(static_cast<bool>(value));
                }
            }
            else if(h5type == HighFive::AtomicType<char>())
            {
                std::vector<char> data = *hdf5util::getAttributeVector<char>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            } 
            else if(h5type == HighFive::AtomicType<unsigned char>())
            {
                std::vector<unsigned char> data = *hdf5util::getAttributeVector<unsigned char>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<short>())
            {
                std::vector<short> data = *hdf5util::getAttributeVector<short>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned short>())
            {   
                std::vector<unsigned short> data = *hdf5util::getAttributeVector<unsigned short>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<int>())
            {   
                std::vector<int> data = *hdf5util::getAttributeVector<int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned int>())
            {   
                std::vector<unsigned int> data = *hdf5util::getAttributeVector<unsigned int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<long int>())
            {   
                std::vector<long int> data = *hdf5util::getAttributeVector<long int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned long int>())
            {   
                std::vector<unsigned long int> data = *hdf5util::getAttributeVector<unsigned long int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<float>())
            {   
                std::vector<float> data = *hdf5util::getAttributeVector<float>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<double>())
            {   
                std::vector<double> data = *hdf5util::getAttributeVector<double>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            } 
            else {
                std::cout << h5type.string() << ": type not implemented. " << std::endl;
            }

        }
        else if(dims.size() == 2)
        {
            // Matrix
            Eigen::MatrixXd mat = *hdf5util::getAttributeMatrix(g, attributeName);
            back[yamlNames.back()] = mat;
        }

        ret = yamlNodes.front();
    }

    return ret;
}

void hdf5MetaTest()
{
    std::string filename = "test.h5";
    auto h5file = hdf5util::open(filename);

    HighFive::Group g = hdf5util::getGroup(h5file, "scanpos0");
    std::vector<size_t> dims = {2000, 3};
    HighFive::DataSpace ds(dims);

    HighFive::DataSetCreateProps properties;
    auto d = hdf5util::createDataset<float>(g, "mydata", ds,  properties);
    
    Transformd mytransform = Transformd::Identity();
    mytransform(0,0) = 2.0;


    YAML::Node meta;
    meta["transform"] = mytransform;
    meta["type"] = "ScanPosition";
    meta["kind"] = "ScanPosition";

    YAML::Node boollist = YAML::Load("[]");
    boollist.push_back(true);
    boollist.push_back(false);
    boollist.push_back(false);
    meta["boollist"] = boollist;

    YAML::Node config;
    config["pose"] = mytransform;
    config["temp"] = 2.0;
    config["bla"] = "hello";


    YAML::Node distortion = YAML::Load("[]");
    for(size_t i = 0; i < 10; i++)
    {
        distortion.push_back(static_cast<double>(i) / 2.0);
    }
    config["distortion"] = distortion;

    YAML::Node config2;
    config2["int0"] = 0; 
    config2["int"] = static_cast<long unsigned int>(10);
    config2["uint"] = static_cast<long int>(-10);
    config2["name"] = "Alex";
    config2["float"] = static_cast<float>(5.5);
    config2["double"] = static_cast<double>(2.2);
    config2["bool"] = false;

    config["conf2"] = config2;
    meta["config"] = config;

    std::cout << "Write Meta to group" << std::endl;
    hdf5util::setAttributeMeta(g, meta);
    std::cout << "Write Meta to dataset" << std::endl;
    hdf5util::setAttributeMeta(*d, meta);

    std::cout << "------------" << std::endl;

    std::cout << "Read Meta from group" << std::endl;
    YAML::Node meta_loaded = hdf5util::getAttributeMeta(g);
    std::cout << "Loaded:" << std::endl;
    std::cout << meta_loaded << std::endl;

    std::cout << "------------" << std::endl;
    std::cout << "Read Meta from dataset" << std::endl;
    YAML::Node meta_loaded2 = hdf5util::getAttributeMeta(*d);
    std::cout << "Loaded: " << std::endl;
    std::cout << meta_loaded2 << std::endl;
}

void hdf5IOTest()
{
    std::string filename = "scan_project.h5";
    HDF5KernelPtr kernel(new HDF5Kernel(filename));
    HDF5SchemaPtr schema(new ScanProjectSchemaRaw(filename));

    descriptions::HDF5IO hdf5io(kernel, schema);

    auto sp = dummyScanProject();
    hdf5io.ScanProjectIO::save(sp);

}

int main(int argc, char** argv)
{
    hdf5IOTest();
    return 0;

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
